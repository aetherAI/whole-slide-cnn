"""ResNet34, ResNet50 w/ and w/o fixup initialization adapted from keras_applications' implementation.

# Reference papers

- [Deep Residual Learning for Image Recognition]
  (https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)
- [Fixup Initialization: Residual Learning Without Normalization]
  (https://arxiv.org/abs/1901.09321) (ICLR 2019)

# Reference implementations

- [ResNet]
  (https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py)

"""

import numpy as np
import os
import tensorflow as tf

from whole_slide_cnn.huge_layer_wrapper import HugeLayerWrapper

BASE_WEIGHTS_PATH = (
    'https://github.com/keras-team/keras-applications/'
    'releases/download/resnet/')
WEIGHTS_HASHES = {
    'resnet50': ('2cb95161c43110f7111970584f804107',
                 '4d473c1dd8becc155b73f8504c6f6626'),
}

## Custom layer definitions
class MixPrecisionCast(tf.keras.layers.Layer):
    def __init__(self, to_fp16, **kwargs):
        super(MixPrecisionCast, self).__init__(**kwargs)
        self.to_fp16 = to_fp16
        self.trainable = False

    def call(self, inputs):
        dtype = tf.float16 if self.to_fp16 else tf.float32
        return tf.cast(inputs, dtype)

    def get_config(self):
        config = {
            "to_fp16":self.to_fp16
        }
        base_config = super(MixPrecisionCast, self).get_config()
        return dict(list(base_config.items()) + list(config.items()) )

    def compute_output_shape(self, input_shape):
        return input_shape

class InitializerScaleWrapper(tf.keras.initializers.Initializer):
    def __init__(self, scale, orig_init):
        super(InitializerScaleWrapper, self).__init__()
        self.orig_init = orig_init
        self.scale = scale

    def __call__(self, shape, dtype=None):
        tensor = self.orig_init(shape=shape, dtype=dtype)
        scaled = tensor * self.scale
        return scaled

    def get_config(self):
        config = {
            "orig_init": tf.keras.initializers.serialize(self.orig_init),
            "scale": self.scale,
        }
        base_config = super(InitializerScaleWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        obj = InitializerScaleWrapper(
            scale=config["scale"],
            orig_init=tf.keras.initializers.deserialize(config["orig_init"]),
        )
        return obj

class ScalarMulAddLayer(tf.keras.layers.Layer):
    '''
        A layer returning wx + b where w and b are scalars.
    '''

    def __init__(self, use_weight=False, use_bias=False, **kwargs):
        super(ScalarMulAddLayer, self).__init__(**kwargs)
        self.use_weight = use_weight
        self.use_bias = use_bias

    def build(self, input_shape):
        if self.use_weight:
            self.weight = self.add_weight(
                name="weight",
                shape=[],
                dtype=tf.float32,
                initializer=tf.keras.initializers.Ones(),
                trainable=True,
            )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[],
                dtype=tf.float32,
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
            )
        self.built = True

    def call(self, inputs, **kwargs):
        x = inputs

        if self.use_weight:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias

        return x

    def get_config(self):
        config = {
            "use_weight": self.use_weight,
            "use_bias": self.use_bias,
        }
        base_config = super(ScalarMulAddLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


## Wrap every layer by HugeLayerWrapper to avoid violating cuDNN limitation of 2G elements.
## If a layer is safe, the behavior of the wrapped layer is same as before.
_use_huge_layer_wrapper = False

def _generate_wrapped_layer(layer):
    global _use_huge_layer_wrapper
    return lambda *args, **kwargs: (
        HugeLayerWrapper(layer(*args, **kwargs)) 
        if _use_huge_layer_wrapper 
        else layer(*args, **kwargs)
    )

def _toggle_use_huge_layer_wrapper(toggle):
    global _use_huge_layer_wrapper
    _use_huge_layer_wrapper = toggle

_Activation = _generate_wrapped_layer(tf.keras.layers.Activation)
_Add = _generate_wrapped_layer(tf.keras.layers.Add)
_BatchNormalization = _generate_wrapped_layer(tf.keras.layers.BatchNormalization)
_Conv2D = _generate_wrapped_layer(tf.keras.layers.Conv2D)
_MaxPooling2D = _generate_wrapped_layer(tf.keras.layers.MaxPooling2D)
_MixPrecisionCast = _generate_wrapped_layer(MixPrecisionCast)
_ScalarMulAddLayer = _generate_wrapped_layer(ScalarMulAddLayer)
_ZeroPadding2D = _generate_wrapped_layer(tf.keras.layers.ZeroPadding2D)

"""
_Activation = lambda *args, **kwargs: HugeLayerWrapper(tf.keras.layers.Activation(*args, **kwargs)) 
_Add = lambda *args, **kwargs: HugeLayerWrapper(tf.keras.layers.Add(*args, **kwargs))
_BatchNormalization = lambda *args, **kwargs: HugeLayerWrapper(tf.keras.layers.BatchNormalization(*args, **kwargs))
_Conv2D = lambda *args, **kwargs: HugeLayerWrapper(tf.keras.layers.Conv2D(*args, **kwargs))
_MaxPooling2D = lambda *args, **kwargs: HugeLayerWrapper(tf.keras.layers.MaxPooling2D(*args, **kwargs))
_MixPrecisionCast = lambda *args, **kwargs: HugeLayerWrapper(MixPrecisionCast(*args, **kwargs))
_ScalarMulAddLayer = lambda *args, **kwargs: HugeLayerWrapper(ScalarMulAddLayer(*args, **kwargs))
_ZeroPadding2D = lambda *args, **kwargs: HugeLayerWrapper(tf.keras.layers.ZeroPadding2D(*args, **kwargs))
"""

def basic_block1(x, filters, kernel_size=3, stride=1,
                 conv_shortcut=True, name=None,
                 norm_use="bn",
                 kernel_initializer='he_normal',
                 kernel_regularizer=None,
                 **kwargs):

    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut is True:
        shortcut = conv_layer(inputs=x, filters=filters, kernel_size=1, strides=stride,
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=kernel_regularizer,
                              name=name + '_0_conv', **kwargs)
        shortcut = normalize_layer(shortcut, norm_use=norm_use, axis=bn_axis, name=name + '_0_')
    else:
        shortcut = x

    x = conv_layer(inputs=x, filters=filters, kernel_size=kernel_size, strides=stride,
                   padding='SAME', name=name + '_1_conv',
                   kernel_regularizer=kernel_regularizer, **kwargs)
    x = normalize_layer(x, norm_use=norm_use, axis=bn_axis, name=name + '_1_')
    x = activation_layer(x, activation='relu', name=name + '_1_relu')
    x = conv_layer(inputs=x, filters=filters, kernel_size=kernel_size, padding='SAME',
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name=name + '_2_conv', **kwargs)
    x = normalize_layer(x, norm_use=norm_use, axis=bn_axis, name=name + '_2_')
    x = _Add(name=name + '_add')([shortcut, x])
    x = activation_layer(x, activation='relu', name=name + "_out")

    return x

def basic_block1_fixup(x, filters, kernel_size=3, stride=1,
                       conv_shortcut=True, name=None,
                       norm_use="bn",
                       kernel_initializer='he_normal',
                       kernel_regularizer=None,
                       num_layers=None,
                       **kwargs):

    assert num_layers != None

    if not isinstance(kernel_initializer, tf.keras.initializers.Initializer):
        kernel_initializer = tf.keras.initializers.get(kernel_initializer)

    scaled_kernel_initializer = InitializerScaleWrapper(
        scale=(num_layers) ** (-0.5),
        orig_init=kernel_initializer,
    )

    original_x = x
    x = _ScalarMulAddLayer(use_bias=True)(x)

    if conv_shortcut is True:
        shortcut = conv_layer(inputs=x, filters=filters, kernel_size=1, strides=stride,
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=kernel_regularizer,
                              name=name + '_0_conv')
    else:
        shortcut = original_x

    x = conv_layer(inputs=x, filters=filters, kernel_size=kernel_size, strides=stride,
                   padding='SAME', name=name + '_1_conv',
                   kernel_initializer=scaled_kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                  )
    x = activation_layer(x, activation='relu', name=name + '_1_relu')

    x = _ScalarMulAddLayer(use_bias=True)(x)
    x = conv_layer(inputs=x, filters=filters, kernel_size=kernel_size, padding='SAME',
                   kernel_initializer='zeros',
                   kernel_regularizer=kernel_regularizer,
                   name=name + '_2_conv')
    x = _ScalarMulAddLayer(use_weight=True, use_bias=True)(x)

    x = _Add(name=name + '_add')([shortcut, x])
    x = activation_layer(x, activation='relu', name=name + "_out")

    return x

def block1(x, filters, kernel_size=3, stride=1,
           conv_shortcut=True, name=None,
           norm_use="bn",
           kernel_initializer='he_normal',
           kernel_regularizer=None,
           **kwargs):

    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
    if conv_shortcut is True:
        shortcut = conv_layer(inputs=x, filters=4 * filters, kernel_size=1, strides=stride,
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=kernel_regularizer,
                              name=name + '_0_conv',
                              **kwargs)
        shortcut = normalize_layer(shortcut, axis=bn_axis, norm_use=norm_use, name=name + '_0_')
    else:
        shortcut = x

    x = conv_layer(inputs=x, filters=filters, kernel_size=1, strides=stride, name=name + '_1_conv',
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   **kwargs)
    x = normalize_layer(x, axis=bn_axis, norm_use=norm_use, name=name + '_1_')
    x = activation_layer(x, activation="relu", name=name+"_1_relu")

    x = conv_layer(inputs=x, filters=filters, kernel_size=kernel_size, padding='SAME',
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name=name + '_2_conv',
                   **kwargs)
    x = normalize_layer(x, axis=bn_axis, norm_use=norm_use, name=name + '_2_')
    x = activation_layer(x, activation="relu", name=name + '_2_relu')

    x = conv_layer(inputs=x, filters=4 * filters, kernel_size=1,
                   name=name + '_3_conv', kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   **kwargs)
    x = normalize_layer(x, axis=bn_axis, norm_use=norm_use, name=name + '_3_')
    x = _Add(name=name + '_add')([shortcut, x])
    x = activation_layer(x, activation="relu", name=name + "_out")

    return x

def block1_fixup(x, filters, kernel_size=3, stride=1,
                 conv_shortcut=True, name=None,
                 norm_use="bn",
                 kernel_initializer='he_normal',
                 kernel_regularizer=None,
                 num_layers=None,
                 **kwargs):
    assert num_layers != None

    if not isinstance(kernel_initializer, tf.keras.initializers.Initializer):
        kernel_initializer = tf.keras.initializers.get(kernel_initializer)

    scaled_kernel_initializer = InitializerScaleWrapper(
        scale=(num_layers) ** (-0.25),
        orig_init=kernel_initializer,
    )

    original_x = x
    x = _ScalarMulAddLayer(use_bias=True)(x)

    if conv_shortcut is True:
        shortcut = conv_layer(inputs=x, filters=4 * filters, kernel_size=1, strides=stride,
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=kernel_regularizer,
                              name=name + '_0_conv',
                              **kwargs)
    else:
        shortcut = original_x

    x = conv_layer(inputs=x, filters=filters, kernel_size=1, strides=stride, name=name + '_1_conv',
                   kernel_initializer=scaled_kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   **kwargs)
    x = activation_layer(x, activation="relu", name=name+"_1_relu")

    x = _ScalarMulAddLayer(use_bias=True)(x)
    x = conv_layer(inputs=x, filters=filters, kernel_size=kernel_size, padding='SAME',
                   kernel_initializer=scaled_kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name=name + '_2_conv',
                   **kwargs)
    x = activation_layer(x, activation="relu", name=name + '_2_relu')

    x = _ScalarMulAddLayer(use_bias=True)(x)
    x = conv_layer(inputs=x, filters=4 * filters, kernel_size=1,
                   name=name + '_3_conv', kernel_initializer='zeros',
                   kernel_regularizer=kernel_regularizer,
                   **kwargs)
    x = _ScalarMulAddLayer(use_weight=True, use_bias=True)(x)

    x = _Add(name=name + '_add')([shortcut, x])
    x = activation_layer(x, activation="relu", name=name + "_out")

    return x

def stack1(x, filters, blocks, stride1=2, name=None,
           norm_use="bn",
           block_type="bottleneck",
           kernel_initializer='he_normal',
           kernel_regularizer=None,
           num_layers=None,
           **kwargs):
    if block_type == 'bottleneck':
        fn = block1
    elif block_type == 'basic_block':
        fn = basic_block1
    elif block_type == 'fixup_bottleneck':
        fn = block1_fixup
    elif block_type == 'fixup_basic_block':
        fn = basic_block1_fixup
    else:
        assert False

    if 'fixup' in block_type:
        assert num_layers != None

    x = fn(x, filters, stride=stride1, name=name + '_block1',
               norm_use=norm_use,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=kernel_regularizer,
               num_layers=num_layers,
               **kwargs)
    for i in range(2, blocks + 1):
        x = fn(x, filters, conv_shortcut=False, name=name + '_block' + str(i),
                   norm_use=norm_use,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   num_layers=num_layers,
                   **kwargs)
    return x

def ResNet(stack_fn,
           preact,
           use_bias,
           model_name='resnet',
           weights='imagenet',
           input_tensor=None,
           input_shape=None,
           norm_use="bn",
           use_mixed_precision=False,
           data_format=None,
           kernel_initializer='he_normal',
           kernel_regularizer=None,
           use_fixup=False,
           use_huge_layer_wrapper=False,
           to_caffe_preproc=False,
           **kwargs):

    _toggle_use_huge_layer_wrapper(use_huge_layer_wrapper)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape=input_shape)
    else:
        if not isinstance(input_tensor, tf.Tensor):
            img_input = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    x = img_input
    if to_caffe_preproc:
        x = tf.reverse(x, axis=[-1]) # RGB -> BGR
        x = (
            x * 127.5 + 
            tf.constant(127.5 - np.array([103.939, 116.779, 123.68]), dtype=tf.float32)
        ) # zero central

    # If data_format is specified, do transpose.
    if data_format != None:
        original_data_format = tf.keras.backend.image_data_format()
        if original_data_format == 'channels_first' and data_format == 'channels_last':
            x = tf.keras.layers.Permute([2, 3, 1])(x)
            tf.keras.backend.set_image_data_format(data_format)
        elif original_data_format == 'channels_last' and data_format == 'channels_first':
            x = tf.keras.layers.Permute([3, 1, 2])(x)
            tf.keras.backend.set_image_data_format(data_format)

    # Invoke mixed precision
    if use_mixed_precision:
        tf.keras.mixed_precision.experimental.set_policy('infer_float32_vars')
        x = _MixPrecisionCast(to_fp16=True)(x)

    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    x = _ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(x)
    x = conv_layer(inputs=x, filters=64, kernel_size=7, strides=2,
                   use_bias=use_bias, name='conv1_conv',
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   **kwargs)

    if preact is False:
        x = normalize_layer(x, axis=bn_axis, norm_use=norm_use, name='conv1_')
        x = activation_layer(x, activation="relu", name='conv1_relu')

    x = _MaxPooling2D(3, strides=2, name='pool1_pool', padding="same")(x)

    pyramid = stack_fn(
        x,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
    )
    pyramid = list(pyramid)
    x = pyramid[-1]

    if preact is True:
        x = normalize_layer(x, axis=bn_axis, norm_use=norm_use, name='post_')
        x = activation_layer(x, activation="relu", name='post_relu')

    # Convert back fp32 for the upcoming global average pooling
    if use_mixed_precision:
        x = _MixPrecisionCast(to_fp16=False)(x)
        tf.keras.mixed_precision.experimental.set_policy('infer')

    # If data_format is specified, transpose back.
    if data_format != None:
        if original_data_format == 'channels_first' and data_format == 'channels_last':
            x = tf.keras.layers.Permute([3, 1, 2])(x)
            tf.keras.backend.set_image_data_format(original_data_format)
        elif original_data_format == 'channels_last' and data_format == 'channels_first':
            x = tf.keras.layers.Permute([2, 3, 1])(x)
            tf.keras.backend.set_image_data_format(original_data_format)

    # Create model.
    model = tf.keras.Model(img_input, x, name=model_name)

    # Load weights.
    if (weights == 'imagenet') and (model_name in WEIGHTS_HASHES):
        file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
        file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = tf.keras.utils.get_file(file_name,
                                            BASE_WEIGHTS_PATH + file_name,
                                            cache_subdir='models',
                                            file_hash=file_hash)
        model.load_weights(weights_path, by_name=True)

    elif (weights == 'imagenet') and (model_name not in WEIGHTS_HASHES):
        raise ValueError("No imagenet pretrain weight for this model.")

    elif weights is not None:
        model.load_weights(weights, by_name=True)

    return model


def ResNet34(weights=None,
             input_tensor=None,
             input_shape=None,
             norm_use="bn",
             use_fixup=False,
             use_huge_layer_wrapper=False,
             **kwargs):

    if use_fixup:
        assert norm_use == None or norm_use == '', (
            'You should not specify normalization layers when using fixup.'
        )

    block_type = 'basic_block' if not use_fixup else 'fixup_basic_block'

    def stack_fn(x, **kwargs):
        C2 = x = stack1(x, 64, 3, stride1=1, name='conv2', norm_use=norm_use,
                        block_type=block_type, num_layers=34, use_bias=False, **kwargs)
        C3 = x = stack1(x, 128, 4, name='conv3', norm_use=norm_use,
                        block_type=block_type, num_layers=34, use_bias=False, **kwargs)
        C4 = x = stack1(x, 256, 6, name='conv4', norm_use=norm_use,
                        block_type=block_type, num_layers=34, use_bias=False, **kwargs)
        C5 = x = stack1(x, 512, 3, name='conv5', norm_use=norm_use,
                        block_type=block_type, num_layers=34, use_bias=False, **kwargs)
        return C2, C3, C4, C5

    return ResNet(stack_fn, False, True, 'resnet34',
                  weights,
                  input_tensor, input_shape,
                  norm_use=norm_use,
                  use_fixup=use_fixup,
                  use_huge_layer_wrapper=use_huge_layer_wrapper,
                  **kwargs)

def ResNet50(weights='imagenet',
             input_tensor=None,
             input_shape=None,
             norm_use="bn",
             use_fixup=False,
             use_huge_layer_wrapper=False,
             **kwargs):

    if use_fixup:
        assert norm_use == None or norm_use == '', (
            'You should not specify normalization layers when using fixup.'
        )

    block_type = 'bottleneck' if not use_fixup else 'fixup_bottleneck'

    def stack_fn(x, **kwargs):
        C2 = x = stack1(x, 64, 3, stride1=1, name='conv2', norm_use=norm_use,
                        block_type=block_type, num_layers=50, **kwargs)
        C3 = x = stack1(x, 128, 4, name='conv3', norm_use=norm_use,
                        block_type=block_type, num_layers=50, **kwargs)
        C4 = x = stack1(x, 256, 6, name='conv4', norm_use=norm_use,
                        block_type=block_type, num_layers=50, **kwargs)
        C5 = x = stack1(x, 512, 3, name='conv5', norm_use=norm_use,
                        block_type=block_type, num_layers=50, **kwargs)
        return C2, C3, C4, C5

    return ResNet(stack_fn, False, True, 'resnet50',
                  weights,
                  input_tensor, input_shape,
                  norm_use=norm_use,
                  use_fixup=use_fixup,
                  use_huge_layer_wrapper=use_huge_layer_wrapper,
                  **kwargs)

def normalize_layer(tensor, name, axis, norm_use='bn'):
    norm_use = norm_use.lower() if norm_use is not None else None # force cast string to lower-case if not None

    if norm_use == "bn":
        x = _BatchNormalization(axis=axis, name=name + 'bn', epsilon=1.001e-5)(tensor)
    elif norm_use == "frozen_bn":
        layer = _BatchNormalization(axis=axis, name=name + 'bn', epsilon=1.001e-5, trainable=False)
        x = layer(tensor, training=False)
    elif norm_use == None or norm_use == "":
        x = tensor
    else:
        assert False, print("Check your norm_use: {} is valid".format(norm_use))
    return x

def conv_layer(inputs, kernel_size, filters, name,
               padding='valid', strides=(1, 1), use_bias=True,
               kernel_initializer="he_normal",
               kernel_regularizer=None,
               **kwargs):
    strides = (strides, strides) if type(strides) == int else strides

    x = _Conv2D(
        kernel_size=kernel_size,
        filters=filters,
        padding=padding,
        strides=strides,
        use_bias=use_bias,
        name=name,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer
    )(inputs)

    return x

def activation_layer(inputs, name, activation="relu"):
    x = inputs
    x = _Activation(activation, name=name)(x)
    return x
