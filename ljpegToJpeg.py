

'''
script to convert entire directories of LJPEG FILES
into plain JPEG files, so they can easily be imported into tensorflow
using tf.image.decode_jpeg()
'''



import subprocess
import os
import fnmatch



currentFilePath = ''
fileNameVar = ''
directForNew = '/Users/Josh/Desktop/BioNeurNets/mamoData/jpegData'
bashCommand = "./ljpeg.py " + currentFilePath + " " + fileNameVar + ".jpg" + " --visual --scale 0.3"

root = '/Users/Josh/Desktop/BioNeurNets/mamoData/'
pattern = "*.LEFT_CC.LJPEG"
#filenames = []

#cd to new directory for new jpeg files
bashCommand1 = 'cd ' + directForNew
process = subprocess.Popen(bashCommand1.split(), stdout=subprocess.PIPE)
output, error = process.communicate()


#go through all files and folders in old directory root and convert to jpeg
#and put in new directory above
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch.fnmatch(name, pattern):
            currentFilePath = os.path.join(path, name)
            fileNameVar = name
            print currentFilePath
            print fileNameVar
            process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE)
            output, error = process.communicate()




'''
OPTIONAL:
Convert to jpeg for visualization with down-sizing scale=0.3 (16-bit TIFF is not good for direct visualization)
./ljpeg.py cases/benigns/benign_01/case0029/C_0029_1.LEFT_CC.LJPEG output.jpg --visual --scale 0.3

'''



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

