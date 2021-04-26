import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Residual block used in ResNet-18, ResNet-34

def block0(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """A residual block.
    Arguments:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      name: string, block label.
    Returns:
      Output tensor for the residual block.
    """
    bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut:
        shortcut = layers.Conv2D(filters, 1, padding='SAME', strides=stride, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, kernel_size, padding='SAME', strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


# ---------------------------------------------------------------------------
# Residual block used in ResNet-50, 101, 152

def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """A residual block.
    Arguments:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      name: string, block label.
    Returns:
      Output tensor for the residual block.
    """
    bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


# ---------------------------------------------------------------------------
# Stack of residual blocks used in ResNet-18, 34

def stack0(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.
    Arguments:
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, stack label.
    Returns:
      Output tensor for the stacked blocks.
    """
    x = block0(x, filters, stride=stride1, name=name + '_block0')
    for i in range(2, blocks + 1):
        x = block0(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x


# ---------------------------------------------------------------------------
# Stack of residual blocks used in ResNet-50, 101, 152

def stack1(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.
    Arguments:
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, stack label.
    Returns:
      Output tensor for the stacked blocks.
    """
    x = block1(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x


# ----------------------------------------------------------------------------
# Gets stacks for specified layer depth

def get_stacks(layer_depth=50):
    stack_dict = {
        16: ((stack0, 1), (stack0, 2), (stack0, 2), (stack0, 2)),
        17: ((stack1, 1), (stack0, 2), (stack0, 2), (stack0, 2)),

        18: ((stack0, 2), (stack0, 2), (stack0, 2), (stack0, 2)),
        19: ((stack1, 1), (stack0, 2), (stack0, 3), (stack0, 2)),
        20: ((stack0, 2), (stack0, 2), (stack0, 3), (stack0, 2)),
        21: ((stack1, 1), (stack0, 2), (stack0, 4), (stack0, 2)),
        22: ((stack0, 2), (stack0, 2), (stack0, 4), (stack0, 2)),
        23: ((stack1, 1), (stack0, 3), (stack0, 4), (stack0, 2)),
        24: ((stack0, 2), (stack0, 3), (stack0, 4), (stack0, 2)),
        25: ((stack1, 1), (stack0, 3), (stack0, 4), (stack0, 3)),
        26: ((stack0, 2), (stack0, 3), (stack0, 4), (stack0, 3)),
        27: ((stack1, 1), (stack0, 3), (stack0, 5), (stack0, 3)),
        28: ((stack0, 2), (stack0, 3), (stack0, 5), (stack0, 3)),
        29: ((stack1, 1), (stack0, 4), (stack0, 5), (stack0, 3)),
        30: ((stack0, 2), (stack0, 4), (stack0, 5), (stack0, 3)),
        31: ((stack1, 1), (stack0, 4), (stack0, 6), (stack0, 3)),
        32: ((stack0, 2), (stack0, 4), (stack0, 6), (stack0, 3)),
        33: ((stack1, 1), (stack0, 5), (stack0, 6), (stack0, 3)),

        34: ((stack0, 3), (stack0, 4), (stack0, 6), (stack0, 3)),
        35: ((stack0, 3), (stack1, 3), (stack0, 6), (stack0, 3)),
        36: ((stack0, 3), (stack0, 4), (stack0, 7), (stack0, 3)),
        37: ((stack0, 3), (stack1, 3), (stack0, 7), (stack0, 3)),
        38: ((stack0, 3), (stack0, 4), (stack0, 7), (stack0, 4)),
        39: ((stack0, 3), (stack1, 3), (stack0, 7), (stack0, 4)),
        40: ((stack0, 4), (stack0, 4), (stack0, 7), (stack0, 4)),
        41: ((stack0, 4), (stack1, 3), (stack0, 7), (stack0, 4)),
        42: ((stack0, 4), (stack0, 4), (stack0, 8), (stack0, 4)),
        43: ((stack0, 4), (stack1, 3), (stack0, 8), (stack0, 4)),
        44: ((stack0, 4), (stack0, 4), (stack0, 9), (stack0, 4)),
        45: ((stack0, 4), (stack1, 3), (stack0, 9), (stack0, 4)),
        46: ((stack0, 4), (stack0, 4), (stack0, 10), (stack0, 4)),
        47: ((stack0, 4), (stack1, 3), (stack0, 10), (stack0, 4)),
        48: ((stack0, 4), (stack0, 4), (stack0, 11), (stack0, 4)),
        49: ((stack0, 4), (stack1, 3), (stack0, 11), (stack0, 4)),

        50: ((stack1, 3), (stack1, 4), (stack1, 6), (stack1, 3)),
        51: ((stack0, 5), (stack1, 4), (stack1, 6), (stack1, 3)),
    }
    
    # 101: ((stack1, 3), (stack1, 4), (stack1, 23), (stack1, 3)),
    # 152: ((stack1, 3), (stack1, 8), (stack1, 36), (stack1, 3)),

    ((conv2_fn, conv2_blocks),
     (conv3_fn, conv3_blocks),
     (conv4_fn, conv4_blocks),
     (conv5_fn, conv5_blocks)) = stack_dict[layer_depth]

    def stack_fn(x):
        x = conv2_fn(x, 64, conv2_blocks, stride1=1, name='conv2')
        x = conv3_fn(x, 128, conv3_blocks, name='conv3')
        x = conv4_fn(x, 256, conv4_blocks, name='conv4')
        return conv5_fn(x, 512, conv5_blocks, name='conv5')

    return stack_fn

    # if layer_depth == 18:
    #     def stack_fn(x):
    #         x = stack0(x, 64, 2, stride1=1, name='conv2')
    #         x = stack0(x, 128, 2, name='conv3')
    #         x = stack0(x, 256, 2, name='conv4')
    #         return stack0(x, 512, 2, name='conv5')
    #
    # if layer_depth == 34:
    #     def stack_fn(x):
    #         x = stack0(x, 64, 3, stride1=1, name='conv2')
    #         x = stack0(x, 128, 4, name='conv3')
    #         x = stack0(x, 256, 6, name='conv4')
    #         return stack0(x, 512, 3, name='conv5')
    #
    # if layer_depth == 50:
    #     def stack_fn(x):
    #         x = stack1(x, 64, 3, stride1=1, name='conv2')
    #         x = stack1(x, 128, 4, name='conv3')
    #         x = stack1(x, 256, 6, name='conv4')
    #         return stack1(x, 512, 3, name='conv5')
    #
    # elif layer_depth == 101:
    #     def stack_fn(x):
    #         x = stack1(x, 64, 3, stride1=1, name='conv2')
    #         x = stack1(x, 128, 4, name='conv3')
    #         x = stack1(x, 256, 23, name='conv4')
    #         return stack1(x, 512, 3, name='conv5')
    #
    # elif layer_depth == 152:
    #     def stack_fn(x):
    #         x = stack1(x, 64, 3, stride1=1, name='conv2')
    #         x = stack1(x, 128, 8, name='conv3')
    #         x = stack1(x, 256, 36, name='conv4')
    #         return stack1(x, 512, 3, name='conv5')
    #
    # return stack_fn


# ----------------------------------------------------------------------------
# ResNet network

def ResNet(
        num_channels=3,  # Number of input color channels in images.
        resolution=32,  # Resolution (h, w) of input images.
        label_size=10,  # Number of labels.
        layer_depth=50,  # ResNet layer depth
        **kwargs):  # Unused keyword args.

    stack_fn = get_stacks(layer_depth)

    img_inputs = keras.Input(shape=(resolution, resolution, num_channels))
    x = layers.Conv2D(64, 7, strides=2, padding='SAME')(img_inputs)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='SAME')(x)
    x = stack_fn(x)
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(label_size, activation='softmax', name='predictions')(x)
    model = keras.Model(inputs=img_inputs, outputs=x, name="resnet_model")

    return model


# ----------------------------------------------------------------------------
# Get layer depth for verification

def get_layer_depth(model):
    layer_names = [l.name for l in model.layers]
    # Filter names to only conv layers, not including skip layer convolutions
    conv_layer_names = [l for l in layer_names if '1_conv' in l or '2_conv' in l or '3_conv' in l]
    return len(conv_layer_names) + 2  # adds top conv layer and max pool
