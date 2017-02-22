

'''
script to convert entire directories of LJPEG FILES
into plain JPEG files, so they can easily be imported into tensorflow
using tf.image.decode_jpeg()

NOTE, B4 running files to convert must be in ljpeg/mamoData

'''



import subprocess
import os
import fnmatch



nORc =''
currentFilePath = ''
fileNameVar = ''
directForNew = 'convertedMamoData/'  #normal or cancer folders


root = '/Users/Josh/PycharmProjects/mamoConvAI/ljpeg/'  #data within /mamoData
root1 = '/Users/Josh/PycharmProjects/mamoConvAI/ljpeg/mamoData'
pattern =  "*.RIGHT_MLO.LJPEG"      # "*.LEFT_CC.LJPEG"    -     "*.LEFT_MLO.LJPEG"   -      "*.RIGHT_CC.LJPEG"
#filenames = []



#cd to new directory for new jpeg files
bashCommand1 = 'cd ' + root
#process = subprocess.Popen(bashCommand1.split(), stdout=subprocess.PIPE)
#output, error = process.communicate()
os.chdir('ljpeg')


process = subprocess.Popen('pwd', stdout=subprocess.PIPE)
output, error = process.communicate()
print output


#go through all files and folders in old directory root and convert to jpeg
#and put in new directory above
for path, subdirs, files in os.walk(root1):
    for name in files:
        if fnmatch.fnmatch(name, pattern):
            currentFilePath = os.path.join(path, name)
            fileNameVar = name
            print currentFilePath[45:]
            print fileNameVar
            if 'cancer' in currentFilePath:
                nORc = 'convertedMamoData/cancerV4/'  #cancer  Vx  v1 = LCC, v2 = lMLO v3 = RCC v4 = rMLO
                bashCommand = "./ljpeg.py " +currentFilePath[
                                              45:] + " " + nORc + fileNameVar + ".jpg" + " --visual --scale 0.3"
                print bashCommand
                process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
            else:
                nORc = 'convertedMamoData/normalV4/'  #normal  Vx  v1 = LCC, v2 = lMLO v3 = RCC v4 = rMLO
                bashCommand = "./ljpeg.py " +currentFilePath[
                                              45:] + " " + nORc + fileNameVar + ".jpg" + " --visual --scale 0.3"
                process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()




'''
OPTIONAL:
Convert to jpeg for visualization with down-sizing scale=0.3 (16-bit TIFF is not good for direct visualization)
./ljpeg.py cases/benigns/benign_01/case0029/C_0029_1.LEFT_CC.LJPEG output.jpg --visual --scale 0.3

'''





