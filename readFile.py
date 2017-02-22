

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import fnmatch
import tensorflow as tf


IMAGE_WIDTH = 100          #changed from 200
IMAGE_HEIGHT = 150          #changed from 350

# Global constants
#start with number currently on my computer......104 images.....even though not enough.......
NUM_CLASSES = 2     # start with 2 classes, normal(0) / irregular(1)
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 561   # will change to 2600
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 70   # want about 10%, so say will change to 200




def readMamo(rsq):

    '''

     Args:
    rsq(random shuffle queue): A queue of strings with the filenames and labels to read from.

    Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result
      width: number of columns in the result
      depth: number of color channels in the result, 1 only for grayscale
      key: a scalar string Tensor describing the filename
      label: an int32 Tensor with the label in the range 0..1.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
    '''


    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()

    # Dimensions of the images, as size reduced and cropped from jpeg originals
    result.height = 150                 # changed from 350
    result.width = 100                   #  changed 200
    result.depth = 1  # changed to 1 from 3 as is greyscale



    # Read a record, getting filenames from the filename_queue.
    gotf, gotl = rsq.dequeue()

    result.key = gotf    # filename string tensor
    result.value = tf.read_file(gotf)  # tenor.string with acutal image
    result.label = gotl  # label int32 0 or 1 tensor



    # decode jpeg
    # downsize and redize to correct tensor size 200X300
    # will want to view actual images in tensorboard later to see how they look
    result.value = tf.image.decode_jpeg(result.value, ratio=8)
    result.value = tf.image.resize_images(result.value,[150,100])  #changed from 350 / 200



    #label, which we convert from uint8->int32.
    result.label = tf.cast(tf.reshape(result.label,[1]), tf.int32)


    # convert image to uint8
    result.uint8image = tf.cast(result.value, tf.uint8)

    return result




def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, rsq, enqueueOP, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 1] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 1] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)

  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)


  return images, tf.reshape(label_batch, [batch_size]), rsq, enqueueOP







def distorted_inputs(data_dir, batch_size):
  """Construct distorted input

  Args:
    data_dir: directory of jpeg images
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 1] size.
    labels: Labels. 1D tensor of [batch_size] size.
    enqueueOP
  """


  #currently only using single LEFT CC view from each case......
  data_dir = '/Users/Josh/PycharmProjects/mamoConvAI/ljpeg/convertedMamoData/train'

  pattern = "*.LEFT_CC.LJPEG.jpg"    #  "*.RIGHT_MLO.LJPEG"    -     "*.LEFT_MLO.LJPEG"   -      "*.RIGHT_CC.LJPEG"
  filenames = []
  labels = []

  # fill both filename list and corresponding label list
  for path, subdirs, files in os.walk(data_dir):
    for name in files:
        if fnmatch.fnmatch(name, pattern):
            ljFileName = os.path.join(path, name)
            filenames.append(ljFileName)
            if 'cancer' in ljFileName:
                labels.append(1)
            else:
                labels.append(0)


  # 104 total converted jpeg images
  num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN


  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  # set shuffle to false for now...... will want to find way to make true later
  # filename_queue = tf.train.string_input_producer(filenames)
  fv = tf.constant(filenames)
  lv = tf.constant(labels)
  rsq = tf.RandomShuffleQueue(1000, 0, [tf.string, tf.int32], shapes=[[], []])



  #create enqueueOP for graph
  enqueueOP = rsq.enqueue_many([fv, lv])


  # Read examples from files in the filename queue.
  read_input = readMamo(rsq)



  reshaped_image = tf.cast(read_input.uint8image, tf.float32)



  height = IMAGE_HEIGHT             #had previously -10
  width = IMAGE_WIDTH               #had previously -10

  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # Randomly crop a [height, width] section of the image.
  distorted_image = tf.random_crop(reshaped_image, [height, width, 1])

  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(distorted_image)

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
 # distorted_image = tf.image.random_contrast(distorted_image,
 #                                            lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(distorted_image)

  # Set the shapes of tensors.
  float_image.set_shape([height, width, 1])
  read_input.label.set_shape([1])





  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size, rsq, enqueueOP,
                                         shuffle=True)




def inputs(eval_data, data_dir, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the Mamo data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, 1] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

  # start by just taking the first image in a case file, there are actually 4 images per case..... could use more later
  #this should be eval data directory
  data_dir = '/Users/Josh/PycharmProjects/mamoConvAI/ljpeg/convertedMamoData/evaluation'

  pattern = "*.LEFT_CC.LJPEG.jpg"
  filenames = []
  labels = []


  for path, subdirs, files in os.walk(data_dir):
      for name in files:
          if fnmatch.fnmatch(name, pattern):
              ljFileName = os.path.join(path, name)
              filenames.append(ljFileName)

              if 'cancer' in ljFileName:
                  labels.append(1)
              else:
                  labels.append(0)


  num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL



  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)



  # Create a queue that produces the filenames and labels to read.
  fv = tf.constant(filenames)
  lv = tf.constant(labels)

  rsqEval = tf.RandomShuffleQueue(200, 0, [tf.string, tf.int32], shapes=[[], []])
  enqueueOPEval = rsqEval.enqueue_many([fv, lv])


  # Read examples from files in the filename queue.
  read_input = readMamo(rsqEval)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)



  height = IMAGE_HEIGHT
  width = IMAGE_WIDTH

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         width, height)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(resized_image)

  # Set the shapes of tensors.
  float_image.set_shape([width, height, 1])
  read_input.label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size, rsqEval, enqueueOPEval,
                                         shuffle=False)
