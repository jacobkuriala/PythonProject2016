# PythonProject2016
Project done for Python FSU CIS 5930 fall 2016

configurationmanager.py contains the global property
ProcessingFolderPath = r'Data/'

Change this location to the location where your data is stored.

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
the intermediate files in the 'ProcessingFolderPath'.

We currently test generate files for only 4 test files in slices and palettes folder
To test original data you can copy all files from Data/sample_data into outer folder
Data/
This was the original data that we used for report so you can test with this to see original results.
