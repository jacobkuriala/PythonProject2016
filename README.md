# PythonProject2016
Project done for Python FSU CIS 5930 fall 2016

Instructions for test run:

configurationmanager.py contains the global property
ProcessingFolderPath = r'Data/'

Change this location to the location where your data is stored.

(referring to the contents of the actual folder Data/ will help understand the following:)
Assumption:
This folder will contain the file file_palette_RGBmapping_rogues.txt which contains a list of file
that must be 'ignored' for processing

The palettes and slices folders within the 'ProcessingFolderPath' will contain the palette files
and their corresponding image file respectively. Both will have the same name.

Note:
All intermediate files are created in the ProcessingFolderPath folder.

If you have all the dependencies then you could simply run lassoregression.py

Note that the intermediate files are not recreated if they exist in the 'ProcessingFolderPath'.
If you wish to refresh the data based on new files in palettes and slices data delete all
the intermediate files in the 'ProcessingFolderPath'(except file_palette_RGBmapping_rogues.txt).

The 'ProcessingFolderPath' currently contains the files generated for the report so if you run
regression you will get the data for all the values. However, the silces and palettes folders 
only contain 4 sample images so if the tests are run then the intermediate files get overwritten with
intermediate files generated for those 4 files.

The original intermediate files generated during the project are located at Data/sample_data. You can 
copy the files from this folder to the parent Data folder ''ProcessingFolderPath'' to get back the 
original values in case you lose them during testing.
