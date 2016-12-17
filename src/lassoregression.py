from __future__ import print_function
from featureio import read_features
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
import scorer
import random
import pallette_manager as pm
import featureio as fio
import clustermanager as cm

import numpy as np
import matplotlib.image as mpimag
import matplotlib.pyplot as plt
import skimage as ski

def get_data():
    feature_dict = read_features()
    palette_feature_list = []
    palette_score_list = []
    for palette_name, feature_list in feature_dict.items():
        palette_feature_list.append(feature_list)
        palette_score_list.append(scorer.getscore(palette_name))

    return palette_feature_list, palette_score_list

def train_model():
    features, scores = get_data()
    X_train, X_test, y_train, y_test = train_test_split(features, scores,
                                                    test_size=0.4,
                                                    random_state=0)
    """
    train_x, test_x, train_y, test_y = cross_validation.train_test_split(
                                        features, scores, test_size=0.2, random_state=42)
    train_x = np.array(train_x)
    test_x = np.array(test_x)
    train_y = np.array(train_y)
    train_y.resize(train_y.size, 1)
    test_y = np.array(test_y)
    test_y.resize(test_y.size,1)

    print( "Dimension of Boston test_x = ", test_x.shape )
    print( "Dimension of test_y = ", test_y.shape )

    print( "Dimension of Boston train_x = ", train_x.shape )
    print( "Dimension of train_y = ", train_y.shape )


    # We scale the inputs to have mean 0 and standard variation 1.

    scaler = preprocessing.StandardScaler()
    #scaler.fit(train_x)
    #train_x = scaler.transform(train_x)
    #test_x  = scaler.transform( test_x )

    # We verify that we have 13 features...

    numFeatures =  train_x.shape[1]

    print( "number of features = ", numFeatures )

    # <h2>Input & Output Place-Holders</h2>
    # Define 2 place holders to the graph, one for the inputs one for the outputs...

    with tf.name_scope("IO"):
        inputs = tf.placeholder(tf.float32, [None, numFeatures], name="X")
        outputs = tf.placeholder(tf.float32, [None, 1], name="Yhat")


    # Define the Coeffs for the Layers
    # For each layer the input vector will be multiplied by a matrix $h$ of dim
    # $n$ x $m$, where $n$ is the dimension of the input vector and $m$ the
    # dimension of the output vector. Then a bias vector of dimension $m$ is added to the product.

    with tf.name_scope("LAYER"):
        # network architecture
        Layers = [numFeatures, 5, 4, 3, 2, 1]

        h1 = rnn.get_weight(Layers[0], Layers[1], "h1", False)
        h2 = rnn.get_weight(Layers[1], Layers[2], "h2", False)
        h3 = rnn.get_weight(Layers[2], Layers[3], "h3", False)
        h4 = rnn.get_weight(Layers[3], Layers[4], "h4", False)
        hout = rnn.get_weight(Layers[4], Layers[5], "hout", False)

        b1 = rnn.get_bias(Layers[1], "b1")
        b2 = rnn.get_bias(Layers[2], "b2")
        b3 = rnn.get_bias(Layers[3], "b3")
        b4 = rnn.get_bias(Layers[4], "b4")
        bout = rnn.get_bias(Layers[5], "bout")


    # Define the Layer operations as a Python function

    def model( inputs, layers ):
        [h1, b1, h2, b2, h3, b3, hout, bout] = layers
        y1 = tf.add( tf.matmul(inputs, h1), b1 )
        y1 = tf.nn.sigmoid( y1 )

        y2 = tf.add( tf.matmul(y1, h2), b2 )
        y2 = tf.nn.sigmoid( y2 )

        y3 = tf.add( tf.matmul(y2, h3), b3 )
        y3 = tf.nn.sigmoid( y3 )

        y4 = tf.add( tf.matmul(y3, h4), b4 )
        y4 = tf.nn.sigmoid( y4 )

        yret  = tf.matmul(y4, hout) + bout
        return yret


    # Define the operations that are performed
    # We define what happens to the inputs (x), when they are provided, and what we do with
    # the outputs of the layers (compare them to the y values), and the type of minimization
    # that must be done.

    with tf.name_scope("train"):
        learning_rate = 0.50
        yout = model( inputs, [h1, b1, h2, b2, h3, b3, hout, bout] )

        cost_op = tf.reduce_mean( tf.pow( yout - outputs, 2 ))
        #cost_op = tf.reduce_sum( tf.pow( yout - outputs, 2 ))
        #cost_op =  tf.reduce_mean(-tf.reduce_sum( yout * tf.log( outputs ) ) )
        #cost_op = tf.reduce_mean(tf.squared_difference(yout, outputs))
        #train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_op)
        #train_op = tf.train.AdamOptimizer( learning_rate=learning_rate ).minimize( cost_op )
        train_op = tf.train.AdagradOptimizer( learning_rate=learning_rate ).minimize( cost_op )


    # Train the Model

    # We are now ready to go through many sessions, and in each one train the
    # model.  Here we train on the whole x-train and y-train data, rather than
    # batching into smaller groups.

    # define variables/constants that control the training
    epoch = 0
    last_cost = 0
    mplt_epochs = 3000
    tolerance = 1e-6

    print( "Beginning Training" )

    sess = tf.Session() # Create TensorFlow session
    with sess.as_default():

        # initialize the variables
        init = tf.initialize_all_variables()
        sess.run(init)

        # start training until we stop, either because we've reached the mplt
        # number of epochs, or successive errors are close enough to each other
        # (less than tolerance)

        costs = []
        epochs= []
        while True:
            # Do the training
            sess.run( train_op, feed_dict={inputs: train_x, outputs: train_y} )

            # Update the user every 1000 epochs
            if epoch % 1000==0:
                cost = sess.run(cost_op, feed_dict={inputs: train_x, outputs: train_y})
                costs.append( cost )
                epochs.append( epoch )

                print( "Epoch: %d - Error: %.4f" %(epoch, cost) )

                # time to stop?
                if epoch > mplt_epochs :
                    # or abs(last_cost - cost) < tolerance:
                    print( "STOP!" )
                    break
                last_cost = cost

            epoch += 1

        # we're done...
        # print some statistics...

        print( "Test Cost =", sess.run(cost_op, feed_dict={inputs: test_x, outputs: test_y}) )

        # compute the predicted output for test_x
        pred_y = sess.run( yout, feed_dict={inputs: test_x, outputs: test_y} )

        print( "\nPrediction\nreal\tpredicted" )
        for (y, yHat ) in list(zip( test_y, pred_y ))[0:10]:
            print( "%1.1f\t%1.1f" % (y, yHat ) )

    r2 =  metrics.r2_score(test_y, pred_y)
    print( "mean squared error = ", metrics.mean_squared_error(test_y, pred_y))
    print( "r2 score (coef determination) = ", metrics.r2_score(test_y, pred_y))
    """
    lasso_model = linear_model.Lasso(alpha=0.1)
    lasso_model.fit(X_train, y_train)

    return lasso_model, X_test, y_test

def generate_row(color_w, beg_color, end_color, steps):
    row = []
    for value in range(beg_color, end_color, steps):
        for x in range(color_w):
            row.append([float(value),float(value),float(value)])

    return row

def generate_palette(h, color_w, beg_color, end_color, steps=10):
    palette = []
    for x in range(1):
        palette.append(generate_row(color_w, beg_color, end_color, steps))
    #print(np.array(palette).shape)
    return np.array(palette, dtype=np.uint8)

def test_model():
    model, x_test, y_test = train_model()
    predicted_scores = model.predict(x_test)
    fig, ax = plt.subplots()
    ax.scatter(y_test, predicted_scores)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=4)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xlabel('Actual', fontsize=18)
    ax.set_ylabel('Predicted',fontsize=18)
    ax.set_title('All features',fontsize=25)
    plt.show()
    print("Coefficient of Determniaiton: " + str(model.score(x_test, y_test)))


def production_model():
    features, scores = get_data()
    palette1 = generate_palette(300,1, 10, 60)
    palette2 = generate_palette(300,100, 205, 255)
    Badpalette = np.array([[[0.0,0.0,0.0],
                            [163.0,73.0,164.0],
                            [205.0,16.0,26.0],
                            [103.0, 152.0,114.0],
                            [63.0, 72.0, 204.0]]],dtype=np.uint8)
    AnotherBadpalette=np.array([[[225.0,225.0,0.0],[0.0,255.0,255.0],[255.0,0.0,255.0],
                                 [0,255,64],[255,0,0]]],dtype=np.uint8)
    goodpalette=np.array([[[221.0,135.0,210.0],[135.0,169.0,221.0],[135.0,221.0,156.0],
                           [218.0,221.0,135.0],[49.0,59.0,79.0]]],dtype=np.uint8)
    #print(features[0])
    #print(goodpalette)
    #print(palette1)
    #print(list(map(lambda x: list(map(float, x)), Badpalette)))

    LAB_palette1 = ski.color.rgb2lab(palette1)
    #print(LAB_palette1.shape)
    #print(LAB_palette1)
    #LAB_palette2 = ski.color.rgb2lab(palette2)
    badpath='/home/er/Downloads/ProjectPyTest/DESIGNSEEDS/DESIGNSEEDS/palettes/bad.png'
    goodpath='/home/er/Downloads/ProjectPyTest/DESIGNSEEDS/DESIGNSEEDS/palettes/good.png'
    lab_bad_palette = ski.color.rgb2lab(Badpalette)[0]
    lab_good_palette = ski.color.rgb2lab(goodpalette)[0]
    lab_bad_palette2 = ski.color.rgb2lab(AnotherBadpalette)[0]
    #lab_bad_paletteu =
    """
    palette_first_row = pm.get_palette_first_row(lab_bad_palette)
    #Badpalette =pm.get_unique_palette_colors(palette_first_row)
    LAB_Badpalette= np.array(pm.get_LAB_palette_from_file(badpath))
    LAB_AnotherBadpalette= ski.color.rgb2lab(AnotherBadpalette)
    LAB_goodpalette=  np.array(pm.get_LAB_palette_from_file(goodpath))
    """
    #p1firstrow = pm.get_palette_first_row(LAB_palette1)
#    p2firstrow = pm.get_palette_first_row(LAB_palette2)
#    pu1 = pm.get_unique_palette_colors(p1firstrow)
#    pu2 = pm.get_unique_palette_colors(p2firstrow)
    #print()
    #print(lab_bad_palette.shape)
    #print(lab_bad_palette)
    #print(LAB_goodpalette)
    lbuart = pm.getArtistPalette('Abovetheclouds610.png')

    Badpalettefeatures = fio.feature_extraction(lbuart,lab_bad_palette)
    #print(Badpalettefeatures)
    Goodpalettefeatures = fio.feature_extraction(lbuart, lab_good_palette)
    AnotherBadpalettefeatures = fio.feature_extraction(lbuart,lab_bad_palette2)
    prod_features = [ Goodpalettefeatures, Badpalettefeatures,AnotherBadpalettefeatures]


    palettea = '/home/er/Downloads/ProjectPyTest/DESIGNSEEDS/DESIGNSEEDS/palettes/AbstractColor.png'

    lab_designer_palette = ski.color.rgb2lab(
                                ski.io.imread(palettea))
    lbfirstrow = pm.get_palette_first_row(lab_designer_palette)
    lbu = pm.get_unique_palette_colors(lbfirstrow)
    kmeans = cm.findimagekmeans('AbstractColor.png')
    #print(lbuart.shape)
    #fio.feature_extraction(lbuart, kmeans)

    #print(type(lbuart))
    #print(LAB_Badpalette)
    #print(pu1.shape)
    pli= plt.imshow(palette2)
    lasso_model = linear_model.Lasso(alpha=0.1)
    lasso_model.fit(features, scores)
    predicted_scores = lasso_model.predict(prod_features)
    print("Predicted: "+str(predicted_scores))
    #print(lbu.shape)
    #print(lasso_model.score())
    #print(lab_designer_palette.shape)
    #print(type(LAB_goodpalette))
    #print(LAB_palette2.shape)
    distanceft = scorer.calculateDistance(lbuart,lab_bad_palette2)
    distancen = scorer.calculateDistance(lbuart, lab_good_palette)
    distancef = scorer.calculateDistance(lbuart,lab_bad_palette)

    print(distancen)
    print(distancef)
    print(distanceft)
    maxdistance = 181/predicted_scores[0]
    print((distancef-distancen)/maxdistance)
    print((predicted_scores[0]-predicted_scores[1])/1)



    #plt.show()



if __name__ == '__main__':
    #print(pm.getArtistPalette('Abovetheclouds610.png'))
    test_model()
