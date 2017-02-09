'''
-convert the LJPEG files to proper number value arrays for each pixel
-also crop to standardize each image to a standard size
-then return image back to image stream....... in main


-use this git repository to get library for ljpeg conversion
https://github.com/aaalgo/ljpeg




A description of ".ics" files:


ics_version 1.0
filename B-3024-1
DATE_OF_STUDY 2 7 1995
PATIENT_AGE 42
FILM
FILM_TYPE REGULAR
DENSITY 4
DATE_DIGITIZED 7 22 1997
DIGITIZER LUMISYS
SELECTED
LEFT_CC LINES 4696 PIXELS_PER_LINE 3024 BITS_PER_PIXEL 12 RESOLUTION 50 OVERLAY
LEFT_MLO LINES 4688 PIXELS_PER_LINE 3048 BITS_PER_PIXEL 12 RESOLUTION 50 OVERLAY
RIGHT_CC LINES 4624 PIXELS_PER_LINE 3056 BITS_PER_PIXEL 12 RESOLUTION 50 OVERLAY
RIGHT_MLO LINES 4664 PIXELS_PER_LINE 3120 BITS_PER_PIXEL 12 RESOLUTION 50 OVERLAY


A description of ".LJPEG" files:

The images have all been stored in a format using LOSSLESS JPEG compression. Even with the compression, each image file
is very large because the films were scanned with a resolution between 42 and 100 microns. The source code for the
program that we used to compress the images is available in the archives at Stanford University. An executable version
for SunOS 5.5 is available on our ftp server ftp://figment.csee.usf.edu/pub/DDSM/software/bin/jpeg. This program is used
 to uncompress the images. Once uncompressed, each image file contains only raw pixel values. Because there is no "header
 information" in the file, the size of each image must be obtained from the ".ics" file.


Inputs:  a filename from filenamequeue



  Returns:
    a cropped image 4600 X 3000 integers


    called in readFile, ReadMamo


'''





def convert(fileName):

    from ljpeg import ljpeg
    import os
    import numpy
    import tensorflow as tf


    #this will convert a single image file within a case directory... start with just first file in each directory
    #    which is left CC
    #must do this for every single case directory to get all images......
    #The loaded is a matrix of type uint16. Typically you want to convert that to float for subsequent processing.

    image = ljpeg.read(fileName).astype('uint8') #2d array of uint8  so correct number of bytes for count...



    #The LJPEG format sometimes has wrong values for width and height (transposed). One has to read the correct values
    #of width and height from the associating .ics file. Below is a sample snippet for this purpose:

    W = None
    H = None
    name = ''

    # find the shape of image

    fullpath = fileName
    directory = os.path.dirname(fullpath)


    for file in os.listdir(directory):
        if file.endswith(".ics"):
            name = file


    counter = 0
    for l in open(name, 'r'):
        counter += 1
        l = l.strip().split(' ')
        if (counter == 11):      #want 11th line for 1st image left CC
            H = int(l[3])  #find 3rd word in line
            W = int(l[5])  #find 5th word in line


    assert W != None
    assert H != None


    if W != image.shape[1]:
        logging.warn('reshape: %s' % path)
    image = image.reshape((H, W))


    # send the converted / standardized image back readFile.py
    # to be used in the image/file stream reader......  don't actually save and replace the image


    # label = N #will have to change this to get from .ics file later for irregular vs regular images


    #crop image to max 4600 rows by 3000 pixels
    image = numpy.asarray(image)
    image = image[0:4599, 0:2999]  #will work if it is a numpy array

    #flatten image to scalar tensor format so is same format as example CIFR images
    image = image.flatten()






    return image





    '''
    OPTIONAL:
    Convert to jpeg for visualization with down-sizing scale=0.3 (16-bit TIFF is not good for direct visualization)
    ./ljpeg.py cases/benigns/benign_01/case0029/C_0029_1.LEFT_CC.LJPEG output.jpg --visual --scale 0.3


    '''




