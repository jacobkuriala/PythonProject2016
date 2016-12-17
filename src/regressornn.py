import tensorflow as tf
import math

def get_weight(m, n, name, relu=True):
    if relu:
        weight = tf.Variable(tf.mul(tf.random_normal([m, n],
                                                     0,
                                                     1/math.sqrt(m),
                                                     dtype=tf.float32,
                                                     name=name),
                                    math.sqrt(2.0/m)
                                    )
                             )
    else:
        weight = tf.Variable(tf.div(tf.random_normal([m, n],
                                                     0,
                                                     1/math.sqrt(m),
                                                     dtype=tf.float32,
                                                     name=name),
                                    math.sqrt(m)
                                    )
                             )

    return weight


def get_bias(m, name):
    #return tf.Variable(tf.zeros([m], dtype=tf.float32, name=name))
    biase = tf.Variable(tf.random_normal([m],
                                          0,
                                         1/math.sqrt(m),
                                          dtype=tf.float32,
                                          name=name)
                         )

    return biase
#'''
