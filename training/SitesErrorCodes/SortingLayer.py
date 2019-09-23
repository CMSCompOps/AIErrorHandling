import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import common_shapes
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
import tensorflow.keras as keras

class SortingLayer(Layer):
  def __init__(self,
               tau,
               axis = 1,
               paxis = 0, #0 means the sorting matrix multiplied from left, 1 from right
               kernel_initializer=keras.initializers.random_uniform(0,0.001),
               kernel_regularizer=None,
               kernel_constraint=keras.constraints.max_norm(0.01),
               name = 'SortingLayer' ,
               FW = None,
               **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)
    super(SortingLayer, self).__init__(**kwargs)

    self.tau = tau
    self.axis = axis
    self.paxis = paxis
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.Name = name
    self.FW = FW
    
  def build(self, input_shape):
    with tf.name_scope( self.Name ):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        print('dtype is:', self.dtype)
        if not (dtype.is_floating or dtype.is_complex):
          raise TypeError('Unable to build `SortingLayer` layer with non-floating point '
                          'dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape)
        rank = len(input_shape)
        if rank <= self.axis :
          raise ValueError('The rank of the input shape is less than the axis of the desired to be sorted in SortingLayer')

        self.dim = tensor_shape.dimension_value(input_shape[self.axis])
        print('dim is:' , self.dim , input_shape)
        self.input_spec = InputSpec(axes={self.axis:self.dim})
        self.sorting_vector = self.add_weight(
          'sorting_vector',
          shape=[self.dim],
          initializer=self.kernel_initializer,
          regularizer=self.kernel_regularizer,
          constraint=self.kernel_constraint,
          dtype=self.dtype,
          trainable=True)

        self.sorting_matrix = self.add_weight(
          'sorting_matrix',
          shape = [self.dim , self.dim],
          dtype=self.dtype,
          trainable = False
          )
        
        self.multiplication_factor = tf.constant( [ [self.dim+1.0-2.0*(i+1)] for i in range(self.dim) ] , name='multipfact_sorting' )
        self.final_transpose_permutation = [ i if i<self.axis else rank-1 if i==self.axis else i-1  for i in range(rank) ]
        print('final_transpose_permutation:',self.final_transpose_permutation)
        self.built = True

  def call(self, inputs):
    with tf.name_scope( self.Name + '_call' ):
        s = tf.stack( [self.sorting_vector]*self.dim )
        A_s = s - tf.transpose(s, perm=[1,0])
        A_s = tf.abs(A_s) 
        B = tf.stack( [ tf.reduce_sum(A_s , 0) ]*self.dim )
        C = gen_math_ops.mat_mul(self.multiplication_factor, tf.convert_to_tensor( [self.sorting_vector] ) )
        P = C - B
        P = tf.nn.softmax(P / self.tau, -1)
        self.sorting_matrix.assign( P )
        inputs = ops.convert_to_tensor(inputs)
        outputs = tf.tensordot( inputs , P , [ [self.axis] , [self.paxis] ] )
        outputs = tf.transpose( outputs , self.final_transpose_permutation )
        return outputs

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
      'tau': self.tau,
      'axis':self.axis,
      'paxis':self.paxis,
      'kernel_initializer': initializers.serialize(self.kernel_initializer),
      'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
      'kernel_constraint': constraints.serialize(self.kernel_constraint)      
    }
    base_config = super(SortingLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))    
