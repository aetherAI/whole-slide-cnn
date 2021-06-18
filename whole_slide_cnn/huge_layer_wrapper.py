from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL


class HugeTensor(object):
    '''Huge tensor
    Huge tensors break the constraint that the size of a tensor cannot exceed 2G.
    It contains a bunch of tensors logically stacked in channel-dimension.

    
    '''

    def __init__(
        self,
        tensor_list,
        broadcastable=None,
    ):
        c_axis = 1 if K.image_data_format() == 'channels_first' else -1

        self.tensor_list = tensor_list
        if broadcastable == None:
            self.broadcastable = (len(tensor_list) == 1) and (tensor_list[0].shape[c_axis].value == 1)
        elif broadcastable == True:
            assert all([tensor.shape[c_axis].value == 1 for tensor in tensor_list])
            self.broadcastable = broadcastable
        else:
            self.broadcastable = broadcastable

    def get_bin_tree_depth(self):
        depth = 0
        width = len(self.tensor_list)
        while width != 1:
            assert width % 2 == 0, 'Invalid tensor_list'
            width /= 2
            depth += 1

        return depth

    @property
    def shape(self):
        c_axis = 1 if K.image_data_format() == 'channels_first' else -1
        shape = list(self.tensor_list[0].shape)
        shape[c_axis] *= len(self.tensor_list)
        return tf.TensorShape(shape)

    @property
    def op(self):
        return self.tensor_list[0].op

    @property
    def dtype(self):
        return self.tensor_list[0].dtype

    def to_tensor(self):
        assert len(self) == 1
        return self.tensor_list[0]

    def __len__(self):
        return len(self.tensor_list)

    @staticmethod
    def from_tensor(tensor):
        return HugeTensor([tensor])

    @staticmethod
    def convert_bin_tree_depth(huge_tensor, bin_tree_depth):
        c_axis = 1 if K.image_data_format() == 'channels_first' else -1

        if huge_tensor.get_bin_tree_depth() > bin_tree_depth:
            assert False, 'Converting to lower bin_tree_depth is not allowed'
        elif huge_tensor.get_bin_tree_depth() == bin_tree_depth:
            return huge_tensor

        depth_diff = bin_tree_depth - huge_tensor.get_bin_tree_depth()
        n_split = 2 ** depth_diff
        if huge_tensor.broadcastable:
            assert all([tensor.shape[c_axis].value == 1 for tensor in huge_tensor.tensor_list])
        else:
            assert all([tensor.shape[c_axis].value % n_split == 0 for tensor in huge_tensor.tensor_list])

        new_tensor_list = []
        for tensor in huge_tensor.tensor_list:
            if huge_tensor.broadcastable:
                new_tensor_list += [tensor] * n_split
            else:
                new_tensor_list += tf.split(tensor, n_split, axis=c_axis)

        return HugeTensor(new_tensor_list, broadcastable=huge_tensor.broadcastable)

class HugeLayerWrapper(KL.Wrapper):
    '''Wrapper to make a layer support huge tensors.

    '''
    MAX_TENSOR_BYTES = 4 * 1024 * 1024 * 1024 - 1
    MAX_WIDTH_X_HEIGHT_FP16 = 16777216 - 1
    BATCH_SIZE = 1
    TYPE_LIST = {
        'simple': [
            'Activation',
            'AlphaDropout',
            'AveragePooling1D',
            'AveragePooling2D',
            'AveragePooling3D',
            'Dropout',
            'ELU',
            'GaussianDropout',
            'GlobalAveragePooling1D',
            'GlobalAveragePooling2D',
            'GlobalAveragePooling3D',
            'GlobalMaxPooling1D',
            'GlobalMaxPooling2D',
            'GlobalMaxPooling3D',
            'LeakyReLU',
            'MaxPooling1D',
            'MaxPooling2D',
            'MaxPooling3D',
            'MixPrecisionCast',
            'PReLU',
            'ReLU',
            'ScalarMulAddLayer',
            'Softmax',
            'SpatialDropout1D',
            'SpatialDropout2D',
            'SpatialDropout3D',
            'ThresholdedReLU',
        ],
        'batch_norm': [
            'BatchNormalization',
        ],
        'group_norm': [
            'GroupNorm',
            'GroupNormV2',
        ],
        'conv': [
            'Conv1D',
            'Conv2D',
            'Conv3D',
        ],
        'merge': [
            'Add',
            'Average',
            'Maximum',
            'Minimum',
            'Multiply',
            'Substract',
        ],
        'padding': [
            'ZeroPadding1D',
            'ZeroPadding2D',
            'ZeroPadding3D',
        ],
        'permute': [
            'Permute',
        ]
    }

    def __init__(self, layer, name=None, **kwargs): 
        if name == None:
            name = layer.name

        super(HugeLayerWrapper, self).__init__(layer, name=name, **kwargs)

        self.layer_type = None
        for layer_type in self.TYPE_LIST:
            if layer.__class__.__name__ in self.TYPE_LIST[layer_type]:
                self.layer_type = layer_type
                break
        assert self.layer_type != None, 'Layer type {} not supported.'.format(layer.__class__.__name__)

        self.c_axis = 1 if K.image_data_format() == 'channels_first' else -1
        self.huge_inputs = None # Original inputs kept by __call__.

    def __call__(self, inputs, **kwargs):
        assert self.MAX_TENSOR_BYTES != None, 'HugeLayerWrapper.set_default should be called before creating a huge layer wrapper.'

        # `inputs` may be a nested list of Tensors and HugeTensors.
        # Since the original __call__ does not recognize HugeTensors, 
        # here we transform all HugeTensors into lists of Tensors.
        # The `inputs` without transforms is copied into self.huge_inputs.
        def copy_nested_list(nested_list, replace_huge_tensor):
            if replace_huge_tensor and isinstance(inputs, HugeTensor):
                return inputs.tensor_list
            if not isinstance(nested_list, list):
                return nested_list

            new_list = []
            for elem in nested_list:
                new_elem = copy_nested_list(elem, replace_huge_tensor)
                new_list.append(new_elem)

            return new_list

        huge_inputs = copy_nested_list(inputs, replace_huge_tensor=False)
        self.huge_inputs = huge_inputs
        inputs = copy_nested_list(inputs, replace_huge_tensor=True)
        res = super(HugeLayerWrapper, self).__call__(inputs, **kwargs)
        if len(res) == 1:
            res = res[0]
        else:
            res = HugeTensor(res)
        self.huge_inputs = None

        return res

    def build(self, input_shape=None):
        def nested_get_shape(nested_input):
            if not isinstance(nested_input, list):
                return nested_input.shape

            new_list = []
            for elem in nested_input:
                new_elem = nested_get_shape(elem)
                new_list.append(new_elem)

            return new_list

        if not self.layer.built:
            input_shape = nested_get_shape(self.huge_inputs)
            self.layer.build(input_shape)
            self.built = True

    def call(self, inputs, **kwargs):
        inputs = self.huge_inputs

        if isinstance(inputs, tf.Tensor):
            inputs = HugeTensor.from_tensor(inputs)
        if isinstance(inputs, list):
            for i, tensor in enumerate(inputs):
                if isinstance(tensor, tf.Tensor):
                    inputs[i] = HugeTensor.from_tensor(tensor)

        def fill_batch_size(tensor):
            input_shape = [dim.value for dim in tensor.shape]
            if input_shape[0] == None:
                input_shape[0] = self.BATCH_SIZE
            assert(all([dim != None for dim in input_shape]))
        if isinstance(inputs, list):
            for tensor in inputs:
                fill_batch_size(tensor)
        else:
            fill_batch_size(inputs)

        # Do computation
        if self.layer_type == 'simple':
            output_tensor_list = self._do_scalar(inputs, **kwargs)
        elif self.layer_type == 'batch_norm':
            output_tensor_list = self._do_batch_norm(inputs, **kwargs)
        elif self.layer_type == 'group_norm':
            output_tensor_list = self._do_group_norm(inputs, **kwargs)
        elif self.layer_type == 'conv':
            output_tensor_list = self._do_conv(inputs, **kwargs)
        elif self.layer_type == 'merge':
            output_tensor_list = self._do_merge(inputs, **kwargs)
        elif self.layer_type == 'padding':
            output_tensor_list = self._do_padding(inputs, **kwargs)
        elif self.layer_type == 'permute':
            print(
                'Warning: Be careful using permute operation along with huge layer wrapper. ' +
                'Either the permute operation do not move channel dimension, or ' +
                'the image format changes immediately after the permute operation.'
            )
            output_tensor_list = self._do_scalar(inputs, **kwargs)

        # Merge
        reduced_tensor_list = self._merge(output_tensor_list)

        return reduced_tensor_list # Return a list of Tensors. It will be packed into HugeTensor in __call__.

    def _get_shape(self, tensor):
        shape = [dim.value for dim in tensor.shape]
        if shape[0] == None:
            shape[0] = self.BATCH_SIZE
        assert(all([dim != None for dim in shape]))
        return shape

    def _merge(self, output_tensor_list):
        tensor_size = max([np.prod(self._get_shape(tensor)) for tensor in output_tensor_list])
        dtype_bytes = 2 if output_tensor_list[0].dtype == tf.float16 else 4
        reduce_grain = 1
        for reduce_grain_candidate in range(len(output_tensor_list), 0, -1):
            if len(output_tensor_list) % reduce_grain_candidate != 0:
                continue
            if reduce_grain_candidate * tensor_size * dtype_bytes <= self.MAX_TENSOR_BYTES:
                reduce_grain = reduce_grain_candidate
                break

        reduced_tensor_list = []
        for i in range(0, len(output_tensor_list), reduce_grain):
            if reduce_grain > 1:
                reduced_tensor = KL.Concatenate(axis=self.c_axis)(output_tensor_list[i: i + reduce_grain])
            else:
                reduced_tensor = output_tensor_list[i]
            reduced_tensor_list.append(reduced_tensor)

        return reduced_tensor_list

    def _get_n_slice(self, shape, dtype, additional_factor=None):
        tensor_size = np.prod(shape)
        dtype_bytes = 2 if dtype == tf.float16 else 4
        n_slice = 1
        while True:
            if shape[self.c_axis] % n_slice != 0:
                n_slice += 1
            elif additional_factor != None and n_slice % additional_factor != 0:
                n_slice += 1
            elif tensor_size // n_slice * dtype_bytes < self.MAX_TENSOR_BYTES:
                break
            elif n_slice == shape[self.c_axis]:
                assert False, 'Tensor too large.'
            else:
                n_slice += 1

        return n_slice

    def _do_scalar(self, inputs, **kwargs):
        output_tensor_list = []
        for input_tensor in inputs.tensor_list:
            output_tensor = self.layer.call(input_tensor, **kwargs)
            output_tensor_list.append(output_tensor)
        return output_tensor_list

    def _do_batch_norm(self, inputs, **kwargs):
        # Calculate the arguments for sliced layers
        n_slices = len(inputs.tensor_list)
        c_size = list(self._get_shape(inputs))[self.c_axis]
        sliced_c_size = int(c_size) // n_slices

        output_tensor_list = []
        for idx, input_tensor in enumerate(inputs.tensor_list):
            # Slicing the underlying layer
            layer_config = self.layer.get_config()
            layer_config["beta_initializer"] = None
            layer_config["gamma_initializer"] = None
            sliced_layer = type(self.layer).from_config(layer_config)

            sliced_layer.build(input_tensor.shape)
            if self.layer.scale:
                sliced_layer.gamma = self.layer.gamma[idx * sliced_c_size: (idx + 1) * sliced_c_size]
            if self.layer.center:
                sliced_layer.beta = self.layer.beta[idx * sliced_c_size: (idx + 1) * sliced_c_size]
            sliced_layer.moving_mean = self.layer.moving_mean[idx * sliced_c_size: (idx + 1) * sliced_c_size]
            sliced_layer.moving_variance = self.layer.moving_variance[idx * sliced_c_size: (idx + 1) * sliced_c_size]

            # Apply
            output_tensor = sliced_layer.call(input_tensor, **kwargs)
            output_tensor_list.append(output_tensor)

        return output_tensor_list

    def _do_group_norm(self, inputs, **kwargs):
        # Calculate the arguments for sliced layers
        n_slices = len(inputs.tensor_list)
        assert self.layer.groups % n_slices == 0, 'Too many slices in the HugeTensor for GroupNorm'
        sliced_n_groups = self.layer.groups // n_slices
        c_size = list(self._get_shape(inputs))[self.c_axis]
        sliced_c_size = int(c_size) // n_slices

        output_tensor_list = []
        for idx, input_tensor in enumerate(inputs.tensor_list):
            # Slicing the underlying layer
            sliced_layer = copy.copy(self.layer)
            sliced_layer.groups = sliced_n_groups
            sliced_layer.beta = tf.slice(self.layer.beta, [idx * sliced_c_size], [sliced_c_size])
            sliced_layer.gamma = tf.slice(self.layer.gamma, [idx * sliced_c_size], [sliced_c_size])

            # Apply
            output_tensor = sliced_layer.call(input_tensor, **kwargs)
            output_tensor_list.append(output_tensor)

        return output_tensor_list

    def _do_conv(self, inputs, **kwargs):
        # Calculate the arguments for sliced layers
        n_input_slices = len(inputs.tensor_list)
        n_output_slices = self._get_n_slice(
            self.layer.compute_output_shape(self._get_shape(inputs)), 
            inputs.tensor_list[0].dtype
        )
        c_size = list(self._get_shape(inputs))[self.c_axis]
        sliced_c_size = int(c_size) // n_input_slices
        output_c_size = self.layer.filters
        sliced_output_c_size = output_c_size // n_output_slices

        # Check if the width x height size violate the limitations when mixed-precision training is 
        # enabled. If that is the case, convert the inputs back to FP32.
        height = list(self._get_shape(inputs))[2] if self.c_axis == 1 else list(self._get_shape(inputs))[1]
        width = list(self._get_shape(inputs))[3] if self.c_axis == 1 else list(self._get_shape(inputs))[2]
        should_stop_mixed_precision = (
            height * width > self.MAX_WIDTH_X_HEIGHT_FP16 and
            inputs.dtype == tf.float16 and
            self.layer.kernel_size == (1, 1) and
            self.layer.strides == (1, 1)
        )
        if should_stop_mixed_precision:
            height_partitions = []
            while np.sum(height_partitions) < height:
                remained = height - np.sum(height_partitions).astype(np.int32)
                max_size = self.MAX_WIDTH_X_HEIGHT_FP16 // width
                partition = np.minimum(remained, max_size)
                height_partitions.append(partition)
        
        # Apply convolution
        output_tensor_list = []
        for output_idx in range(n_output_slices):
            # Apply convolution seperately on each input slice
            partial_sum_list = []
            for input_idx, input_tensor in enumerate(inputs.tensor_list):
                use_bias = self.layer.use_bias
                
                # Slicing the underlying layer
                layer_config = self.layer.get_config()
                layer_config["filters"] = sliced_output_c_size
                layer_config["use_bias"] = False # Will add this back in the end
                layer_config["kernel_initializer"] = None
                layer_config["bias_initializer"] = None
                sliced_layer = type(self.layer).from_config(layer_config)
                sliced_layer.build(input_tensor.shape)
                sliced_layer.kernel = tf.slice(
                    self.layer.kernel,
                    [0, 0, input_idx * sliced_c_size, output_idx * sliced_output_c_size],
                    [-1, -1, sliced_c_size, sliced_output_c_size]
                )

                # Apply
                if should_stop_mixed_precision:
                    input_tensors = tf.split(
                        input_tensor, 
                        height_partitions, 
                        axis=(2 if self.c_axis == 1 else 1),
                    )
                    output_tensors = []
                    for input_tensor_partition in input_tensors:
                        output_tensor_partition = sliced_layer.call(input_tensor_partition, **kwargs)
                        output_tensors.append(output_tensor_partition)
                    output_tensor = tf.concat(
                        output_tensors, 
                        axis=(2 if self.c_axis == 1 else 1),
                    )
                else:
                    output_tensor = sliced_layer.call(input_tensor, **kwargs)

                partial_sum_list.append(output_tensor)

            # Add bias
            if use_bias:
                sliced_bias = tf.slice(
                    self.layer.bias,
                    [output_idx * sliced_output_c_size],
                    [sliced_output_c_size]
                )

                broadcast_shape = [1] * len(self._get_shape(output_tensor))
                broadcast_shape[self.c_axis] = sliced_output_c_size
                sliced_bias = K.reshape(
                    sliced_bias, 
                    broadcast_shape
                )

            # Aggregation
            if any([tensor.dtype == tf.float16 for tensor in partial_sum_list]):
                # Mixed precision
                for idx, tensor in enumerate(partial_sum_list):
                    if tensor.dtype == tf.float16:
                        fp32_tensor = K.cast(tensor, tf.float32)
                        partial_sum_list[idx] = fp32_tensor
                output_tensor = tf.add_n(partial_sum_list)
                if use_bias:
                    sliced_bias = K.cast(sliced_bias, tf.float32)
                    output_tensor = output_tensor + sliced_bias    
                output_tensor = K.cast(output_tensor, tf.float16)
            else:
                output_tensor = tf.add_n(partial_sum_list)
                if use_bias:
                    output_tensor = output_tensor + sliced_bias    

            output_tensor_list.append(output_tensor)

        return output_tensor_list

    def _do_merge(self, inputs, **kwargs):
        # Unify the binary tree depth of each huge tensor in inputs
        unified_inputs = []
        target_bin_tree_depth = max([huge_tensor.get_bin_tree_depth() for huge_tensor in inputs])
        for idx, huge_tensor in enumerate(inputs):
            if huge_tensor.get_bin_tree_depth() != target_bin_tree_depth:
                unified_huge_tensor = HugeTensor.convert_bin_tree_depth(huge_tensor, target_bin_tree_depth)
            else:
                unified_huge_tensor = huge_tensor
            unified_inputs.append(unified_huge_tensor)

        output_tensor_list = []
        for idx in range(2 ** target_bin_tree_depth):
            output_tensor = self.layer.call([huge_tensor.tensor_list[idx] for huge_tensor in unified_inputs], **kwargs)
            output_tensor_list.append(output_tensor)
        return output_tensor_list
        
    def _do_padding(self, inputs, **kwargs):
        n_input_slices = len(inputs.tensor_list)
        n_output_slices = self._get_n_slice(
            self.layer.compute_output_shape(self._get_shape(inputs)), 
            inputs.tensor_list[0].dtype,
            n_input_slices,
        )
        n_further_slices = 1 if n_input_slices >= n_output_slices else n_output_slices // n_input_slices
        assert all([tensor.shape[self.c_axis].value % n_further_slices == 0 for tensor in inputs.tensor_list])

        output_tensor_list = []
        for input_tensor in inputs.tensor_list:
            if n_further_slices > 1:
                splited_tensor_list = tf.split(input_tensor, n_further_slices, axis=self.c_axis)
                for splited_tensor in splited_tensor_list:
                    output_tensor = self.layer.call(splited_tensor, **kwargs)
                    output_tensor_list.append(output_tensor)
            else:
                output_tensor = self.layer.call(input_tensor, **kwargs)
                output_tensor_list.append(output_tensor)
        return output_tensor_list

    @property
    def trainable_weights(self):
        return self.layer.trainable_weights

    @staticmethod
    def set_default(
        max_tensor_bytes=(4 * 1024 * 1024 * 1024 - 1), 
        batch_size=1
    ):
        HugeLayerWrapper.MAX_TENSOR_BYTES = max_tensor_bytes
        HugeLayerWrapper.BATCH_SIZE = batch_size

    @staticmethod
    def get_supported_layer_list():
        layer_list = []
        for type in HugeLayerWrapper.TYPE_LIST:
            for layer in HugeLayerWrapper.TYPE_LIST[type]:
                layer_list.append(layer)
        return layer_list

