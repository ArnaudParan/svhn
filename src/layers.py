import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras import activations
from keras import utils

class ShapeChanger(Layer):

    def __init__(self, **kwargs):
        self.output_dim = 66
        super(ShapeChanger, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(ShapeChanger, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        bs = K.shape(x)[0]
        return K.concatenate([x[:,:7], tf.fill(tf.stack([bs, 4]), 0.0), x[:,7:]])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {}
        return config

class FeatureExtractor(Layer):

    def __init__(self, data_format=None, activation=None, **kwargs):
        self.activation = activations.get(activation)
        self.data_format = K.normalize_data_format(data_format)
        self.padding = utils.normalize_padding("valid")
        super(FeatureExtractor, self).__init__(**kwargs)

    def build(self, input_shape):
        super(FeatureExtractor, self).build(input_shape)

    def call(self, inputs):
        # TODO split the data correctly
        outputs = K.conv2d(
                inputs,
                self.kernel,
                strides=1,
                padding=self.padding,
                dilatation_rate=1)

        if self.activation is not None:
            return self.activation(outputs)

    def _compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = utils.conv_output_length(
                        space[i],
                        self.kernel_size[i],
                        padding=self.padding,
                        stride=self.strides[i],
                        dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                            [self.filters])
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = utils.conv_output_length(
                        space[i],
                        self.kernel_size[i],
                        padding=self.padding,
                        stride=self.strides[i],
                        dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0], self.filters] +
                                            new_space)


    def get_config(self):
        config = {
            "data_format": self.data_format,
            "activation": activations.serialize(self.activation)
        }

        base_config = super(FeatureExtractor, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
