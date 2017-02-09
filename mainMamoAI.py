
'''
Simple convolutional neural networks project using tensor flow

    identify lesions versus no leasons normal mamograms

    use large public USF database with 2600 or so images to do so
        -https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM;jsessionid=B24B6C295F583A40EBBF6FD262E84957
        -http://marathon.csee.usf.edu/Mammography/Database.html

        -first files are downloaded, but are in LJPEG format..... use ljpeg github python library for these..........



files:
    -main
    -readFile
    -buildModel
    -trainModel
    -evalModel

generally follows the tensor flow tutorial on conv neur nets
    -https://www.tensorflow.org/tutorials/deep_cnn/


remote github repository:  MamConvNet


'''




import tensorflow as tf
import numpy
import math








