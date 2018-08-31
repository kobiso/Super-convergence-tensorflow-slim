from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.util.tf_export import tf_export
import tensorflow as tf

@tf_export("train.cyclic_learning_rate")
def cyclic_learning_rate(global_step,
                         learning_rate=0.01,
                         max_lr=0.1,
                         step_size=50000.,
                         gamma=0.99994,
                         max_steps=100000.,
                         scale_rate=0.9,
                         mode='triangular',
                         policy=None,
                         name=None):
  """Cyclic learning rate (CLR).
  This method is revised from [TensorFlow pull request: Add support for Cyclic Learning Rate](https://github.com/tensorflow/tensorflow/pull/20758)
  From the paper:
  Smith, Leslie N. "Cyclical learning
  rates for training neural networks." 2017.
  [https://arxiv.org/pdf/1506.01186.pdf]
  This method lets the learning rate cyclically
  vary between reasonable boundary values
  achieving improved classification accuracy and
  often in fewer iterations.
  This code varies the learning rate linearly between the
  minimum (learning_rate) and the maximum (max_lr).
  It returns the cyclic learning rate. It is computed as:
  ```python
  cycle = floor( 1 + global_step / ( 2 * step_size ) )
  x = abs( global_step / step_size – 2 * cycle + 1 )
  clr = learning_rate + ( max_lr – learning_rate ) * max( 0 , 1 - x )
  ```
  Modes:
    'triangular':
      Default, linearly increasing then linearly decreasing the
      learning rate at each cycle.
    'triangular2':
      The same as the triangular policy except the learning
      rate difference is cut in half at the end of each cycle.
      This means the learning rate difference drops after each cycle.
    'exp_range':
      The learning rate varies between the minimum and maximum
      boundaries and each boundary value declines by an exponential
      factor of: gamma^global_step.
  Args:
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
      Global step to use for the cyclic computation.  Must not be negative.
    learning_rate: A scalar `float32` or `float64` `Tensor` or a
    Python number.  The initial learning rate which is the lower bound
      of the cycle (default = 0.1).
    max_lr:  A scalar. The maximum learning rate boundary.
    step_size: A scalar. The number of iterations in half a cycle.
      The paper suggests step_size = 2-8 x training iterations in epoch.
    gamma: constant in 'exp_range' mode:
      gamma**(global_step)
    max_steps: A scalar. The number of total iterations.
    scale_rate: A scale factor for decreasing the learning rate after the completion of one cycle.
      Must be between 0 and 1.
    mode: one of {triangular, triangular2, exp_range}.
        Default 'triangular'.
        Values correspond to policies detailed above.
    policy: one of {None, one-cycle}.
        Default 'None'.
    name: String.  Optional name of the operation.  Defaults to
      'CyclicLearningRate'.
  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The cyclic
    learning rate.
  Raises:
    ValueError: if `global_step` is not supplied.
  """

  if global_step is None:
    raise ValueError("global_step is required for cyclic_learning_rate.")

  with ops.name_scope(name, "CyclicLearningRate",
                      [learning_rate, global_step]) as name:
    learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
    dtype = learning_rate.dtype
    global_step = math_ops.cast(global_step, dtype)
    step_size = math_ops.cast(step_size, dtype)
    max_steps = math_ops.cast(max_steps, dtype)

    def cyclic_lr():
      """Helper to recompute learning rate; most helpful in eager-mode."""
      # computing: cycle = floor( 1 + global_step / ( 2 * step_size ) )
      double_step = math_ops.multiply(2., step_size)
      global_div_double_step = math_ops.divide(global_step, double_step)
      cycle = math_ops.floor(math_ops.add(1., global_div_double_step))

      # computing: x = abs( global_step / step_size – 2 * cycle + 1 )
      double_cycle = math_ops.multiply(2., cycle)
      global_div_step = math_ops.divide(global_step, step_size)
      tmp = math_ops.subtract(global_div_step, double_cycle)
      x = math_ops.abs(math_ops.add(1., tmp))

      # computing: clr = learning_rate + ( max_lr – learning_rate ) * max( 0, 1 - x )
      a1 = math_ops.maximum(0., math_ops.subtract(1., x))
      a2 = math_ops.subtract(max_lr, learning_rate)
      clr = math_ops.multiply(a1, a2)

      if mode == 'triangular2':
        clr = math_ops.divide(clr, math_ops.cast(math_ops.pow(2, math_ops.cast(
            cycle-1, tf.int32)), tf.float32))
      if mode == 'exp_range':
        clr = math_ops.multiply(math_ops.pow(gamma, global_step), clr)

      return math_ops.add(clr, learning_rate, name=name)
    
    def after_cycle():
      gap = math_ops.subtract(global_step, math_ops.multiply(2., step_size))
      cur_percent = math_ops.divide(gap, math_ops.subtract(max_steps, math_ops.multiply(2., step_size)))
      temp = math_ops.add(1., math_ops.multiply(cur_percent, -0.99))
      next_lr = math_ops.multiply(learning_rate, math_ops.multiply(temp, scale_rate))
      
      return next_lr
    
    if policy == 'one_cycle':
      cyclic_lr = tf.cond(tf.less(global_step, 2*step_size), cyclic_lr , after_cycle)
    else:
      cyclic_lr = cyclic_lr()

    return cyclic_lr
  
def cyclical_momentum(global_step,
                      min_momentum=0.85,
                      max_momentum=0.95,
                      step_size=50000.,
                      name=None):
  """Cyclical momentum
  Args:
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
      Global step to use for the cyclic computation.  Must not be negative.
    step_size: A scalar. The number of iterations in half a cycle.
      The paper suggests step_size = 2-8 x training iterations in epoch.
    min_momentum:  A scalar. The minimum learning rate boundary.
    max_momentum:  A scalar. The maximum learning rate boundary.  
    name: String.  Optional name of the operation.  Defaults to 'Learning_rate_range_test'.
  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.
  Raises:
    ValueError: if `global_step` is not supplied.
  Reference:
    A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay by       Leslie N. Smith (https://arxiv.org/abs/1803.09820)
  """
  
  if global_step is None:
    raise ValueError("global_step is required for cyclic_momentum.")
  with ops.name_scope(name, "CyclicMomentum", [global_step]) as name:
    min_momentum = ops.convert_to_tensor(min_momentum, name="min_momentum")
    max_momentum = ops.convert_to_tensor(max_momentum, name="max_momentum")
    dtype = min_momentum.dtype
    global_step = math_ops.cast(global_step, dtype)
    step_size = math_ops.cast(step_size, dtype)
      
    def first_cycle():  
      # computing: global_step / step_size
      cur_percent = math_ops.divide(global_step, step_size)

      # computing: max_momentum - cur_percent * (max_momentum - min_momentum)
      gap_momentum = math_ops.subtract(max_momentum, min_momentum)
      new_momentum = math_ops.subtract(max_momentum, math_ops.multiply(cur_percent, gap_momentum))
      
      return new_momentum
    
    def second_cycle():
      # computing: 1 - ((global_step - step_size) / step_size)
      gap = math_ops.subtract(global_step, step_size)        
      cur_percent = math_ops.subtract(1., math_ops.divide(gap, step_size))

      # computing: max_momentum - cur_percent * (max_momentum-min_momentum)
      gap_momentum = math_ops.subtract(max_momentum, min_momentum)        
      new_momentum = math_ops.subtract(max_momentum, math_ops.multiply(cur_percent, gap_momentum))
      
      return new_momentum

    # global_step < step_size
    def which_cycle(): return tf.cond(tf.less(global_step, step_size), first_cycle, second_cycle)
    # global_step > 2*step_size
    def after_cycle(): return max_momentum
      
    # global_step < 2*step_size
    next_momentum = tf.cond(tf.less(global_step, 2*step_size), which_cycle, after_cycle)

    return next_momentum
  
def learning_rate_range_test(global_step,
                             max_steps=100000,
                             min_lr=0.00001,
                             max_lr=3.0,
                             name=None):
  """Learning rate range test
  Args:
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
      Global step to use for the cyclic computation.  Must not be negative.
    max_steps: A scalar. The number of total iterations.
    min_lr:  A scalar. The minimum learning rate boundary.
    max_lr:  A scalar. The maximum learning rate boundary.  
    name: String.  Optional name of the operation.  Defaults to 'Learning_rate_range_test'.
  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.
  Raises:
    ValueError: if `global_step` is not supplied.
  Reference:
    Cyclical Learning Rates for Training Neural Networks by Leslie N. Smith (https://arxiv.org/abs/1506.01186)
  """

  if global_step is None:
    raise ValueError("global_step is required for learning_rate_range_test.")

  with ops.name_scope(name, "Learning_rate_range_test", [global_step]) as name:
    inc_rate = (max_lr - min_lr) / max_steps
    next_lr = min_lr + inc_rate * tf.cast(global_step, tf.float32)

  return next_lr