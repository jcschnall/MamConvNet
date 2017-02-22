from math import sqrt
import tensorflow as tf

def put_kernels_on_grid (kernel, pad = 1):

    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''
    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: print('Who would enter a prime number of filters')
                return (i, int(n / i))
    (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
    print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scaling to [0, 255] is not necessary for tensorboard
    return x7

    '''
    This won't work for outputting second convolution, as last dimesnion of output must be 1,3, or 4 only for num channels
    not 10 whatever number of output channels i have in second kernel.   Will need to furthur subdivide the second convolution
    output somehow, to get the dimensions of the data down, unsure how to do this.............................

    something about this code is not right at all, number of features/output channels computed, should not be transposed
    to be the num channels output here / which is only for distinguishing between color/ vs black and white, (1,3, or 4)
    the input channels data should be used in the convolution to help find new features...... just doesn't add up here.......


    '''


#
# ... and somewhere inside "def train():" after calling "inference()"
#

# Visualize conv1 features
#with tf.variable_scope('conv1'):
#  tf.get_variable_scope().reuse_variables()
#  weights = tf.get_variable('weights')
#  grid = put_kernels_on_grid (weights)
#  tf.image_summary('conv1/features', grid, max_images=1)


'''
Hi jeandut

Here is a snippet of code fo visualising convolutional layer, in my case it's of shape [5, 5, 1, 24] so you will have to adapt it to your case.
 I reformatted the 5x5 24-deep convolution kernel as a square grid of 25 5x5 images, the last one being just padding:

W1_a = W1                            # [5, 5, 1, 24]
W1pad= tf.zeros([5, 5, 1, 1])        # [5, 5, 1, 1]
W1_b = tf.concat(3, [W1_a, W1pad])   # [5, 5, 1, 25]
W1_c = tf.split(3, 25, W1_b)         # 25 x [5, 5, 1, 1]
W1_row0 = tf.concat(0, W1_c[0:5])    # [25, 5, 1, 1]
W1_row1 = tf.concat(0, W1_c[5:10])   # [25, 5, 1, 1]
W1_row2 = tf.concat(0, W1_c[10:15])  # [25, 5, 1, 1]
W1_row3 = tf.concat(0, W1_c[15:20])  # [25, 5, 1, 1]
W1_row4 = tf.concat(0, W1_c[20:25])  # [25, 5, 1, 1]
W1_d = tf.concat(1, [W1_row0, W1_row1, W1_row2, W1_row3, W1_row4]) # [25, 25, 1, 1]
W1_e = tf.reshape(W1_d, [1, 25, 25, 1])
Wtag = tf.placeholder(tf.string, None)
tf.image_summary(Wtag, W1_e)


'''

