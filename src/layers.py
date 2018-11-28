from keras import backend as K
from keras.engine.topology import Layer

class ShapeChanger(Layer):

    def __init__(self, **kwargs):
        self.output_dim = 66
        super(ShapeChanger, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(ShapeChanger, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.concatenate([x[:7], K.constant([0. for _ in range(4)]), x[7:]])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {}
        return config
