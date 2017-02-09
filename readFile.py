

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import fnmatch
# from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import ljpegConvert
import copy



#x by x pixels each image
#for LJPEG...... need to look at each case one at a time...... and read .ics file
#.ics file lists number of lines and number of pixels per line for each single image
#would have to crop each image to a uniform standard... 4600 lines X 3000 pixels or so

# the formate of CIFAR-10 data below
# data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain
# the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the
# first 32 entries of the array are the red channel values of the first row of the image.
# labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.



#IMAGE_SIZE = 4600
IMAGE_WIDTH = 4600
IMAGE_HEIGHT = 3000

# Global constants
#start with 2 classes, normal(N) / irregular(I)
#only 2600 cases or images to compare
#start with number currently on my computer......104 images.....even though not enough.......
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 104
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 104   # as currently using same data, will change later......




def readMamo(filename_queue):

    '''

     Args:
    filename_queue: A queue of strings with the filenames to read from.

    Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result
      width: number of columns in the result
      depth: number of color channels in the result
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
    '''


    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()

    # Dimensions of the images .
    label_bytes = 1  # N or I..... will get the label from reading directory
    result.height = 4600   #standardized sizes..... so will slightly crop images......
    result.width = 3000
    result.depth = 1  # changed to 1 from 3 as is grayscale
    image_bytes = result.height * result.width * result.depth


    # Read a record, getting filenames from the filename_queue.
    reader = tf.WholeFileReader()


    result.key, result.value = reader.read(filename_queue) # here key is the filename and value is the whole file


    '''
    SCREW THIS:
    GOING TO HAVE TO JUST WRITE A PYTHONG SCRIPT TO CONVERT ALL LJPEG IMAGES
    TO JPEG IMAGES, BEFORE EVEN RUNNING TENSORFLOW AT ALL...............

    '''



    # call ljpeg for conversion here, using duplicate Queue file_name
    result.value = ljpegConvert.convert(result.key)


    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = result.value   #tf.decode_raw(value, tf.uint8)



    # The first bytes represent the label, which we convert from uint8->int32.
    #result.label = tf.cast(
    #    tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    #take directory from filename to check if normal(1) or irregular(2)
    if "cancer" in result.key:
        result.label = 2

    else:
        result.label = 1


    # The remaining bytes represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(record_bytes,
        [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result




def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
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
  tf.image_summary('images', images)

  '''
  below isn't necessary as ljpeg is converted in above function, and images are already in correct form
  run command in ljpeg directory and will output viewable file to output.jpg
  ./ljpeg.py /Users/Josh/Desktop/BioNeurNets/normal_10/case3660/B_3660_1.LEFT_CC.LJPEG output.jpg --visual --scale 0.3
  '''

  return images, tf.reshape(label_batch, [batch_size])







def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """


  # start by just taking the first image in a case file, there are actually 4 images per case..... could use more later
  root = '/Users/Josh/Desktop/BioNeurNets/mamoData'
  pattern = "*.LEFT_CC.LJPEG"
  filenames = []

  for path, subdirs, files in os.walk(root):
      for name in files:
          if fnmatch.fnmatch(name, pattern):
              ljFileName = os.path.join(path, name)
              filenames.append(ljFileName)


  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)



  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = readMamo(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_HEIGHT
  width = IMAGE_WIDTH

  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # Randomly crop a [height, width] section of the image.
  #this will have to take images before uniform cropping, for it to work............will have to change
  distorted_image = tf.random_crop(reshaped_image, [height, width, 1])

  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(distorted_image)

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(distorted_image)

  # Set the shapes of tensors.
  float_image.set_shape([height, width, 1])
  read_input.label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
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
  data_dir = '/Users/Josh/Desktop/BioNeurNets/mamoData'
  pattern = "*.LEFT_CC.LJPEG"
  filenames = []


  if not eval_data:
      for path, subdirs, files in os.walk(data_dir):
          for name in files:
              if fnmatch(name, pattern):
                  ljFileName = os.path.join(path, name)
                  filenames.append(ljFileName)


      num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN


  else:
    filenames = [os.path.join(data_dir, 'test_batch.bin')]  # will need to implement this later
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL




  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  #set shuffle to false for now...... will want to find way to make true later
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = readMamo(filename_queue)
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
                                         min_queue_examples, batch_size,
                                         shuffle=False)
