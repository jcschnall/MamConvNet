
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

'''
This module will integrate all different views together...... as in train different views one at a time
then tage an aggregate score of all views to make decision of class....................................

0) convert images for other views and make new folders..................

1)train each view separately , so 4 instances of train model with differnt file sources, and different folders for
the model output.........., would essentially just run the training program 4 times in a row with different flag options

2)for the eval, send one file from each view into each of the 4 different models and compute a score for each model
 so need 4 paralell models, and drawing on 4 different files sources................

3)add up this score to get an over all classification of the 4 images


-so need 4 different instances of the training, and then 4 different instances of eval, and add evals scores for aggregate score
'''




def main():
    globals(view)
    view = input("Enter the view you would like to train (1-4) or enter 5 to evaluate: ")
    if(view == 1)
        trainModel.py -flag1  - flag2 - flag3
    elif(view == 2)
        trainModel.py - flag1 - flag2 - flag3
    elif(view == 3)
        trainModel.py - flag1 - flag2 - flag3
    elif(view == 4)
        trainModel.py - flag1 - flag2 - flag3
    elif(view ==5)
        evalModel.py - flag - flag


    return 0






















