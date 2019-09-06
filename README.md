### keras
---
https://github.com/keras-team/keras


```py
// keras/engine/input_layer.py

class InputLayer(Layer):
  """
  """
  @interfaces.legacy_input_support
  def __init__(self, input_shape=None, batch_size=None,
      batch_input_shape=None,
      dtype=None, input_tensor=None, sparse=False, name=None):
    if not name:
      prefix = 'input'
      name = prefix + '_' + str(K.get_uid(prefix))
    super(InputLayer, self).__init__(dtype=dtype, name=name)
    
    self.trainable = False
    self.built = True
    self.sparse = sparse
    self.supports_masking = True
    
    if input_shape and batch_input_shaped:
      raise ValueError('Only provide the input_shape OR '
        'batch_input_shape arguemnt to '
        'InputLayer, not both at the same time.')
    if input_tensor is not None and batch_input_shape is None:
      try:
        batch_input_shape = K.int_shape(input_tensor)
      except TypeError:
        if not input_shape and not batch_input_shape:
          raise ValueError('InputLayer was provided'
            'an input_tensor argument, '
            'but its inputshape cannot be '
            'automatically inferred. '
            'You should pass an input_shape or '
            'batch_input_shape argument.')
    if not batch_input_shape:
      if not input_shape:
        raise ValueError('An Input layer should be passed either '
          'a `batch_input_shape` or an `input_shape`.')
      else:
        batch_input_shape = (batch_size,) + tuple(input_shape)
    else:
      batch_input_shape = tuple(batch_input_shape)
  else:
    batch_input_shape = tuple(batch_input_shape)
    
  if not dtype:
    if input_tensor is None:
      dtype = K.floatx()
    else:
      dtype = K.dtype(input_tensor)
      
  self.batch_input_shape = batch_input_shape
  self.dtype = dtype
  
  if input_tensor is None:
    self.is_placeholder = True
    input_tensor = K.placeholder(shape=batch_input_shape,
        dtype=dtype,
        sparse=self.sparse,
        name=self.name)
  else:
    self.is_placeholder = False
    input_tensor._keras_shape = batch_input_shape
    
  input_tensor._uses_learning_phase = False
  input_tensor._keras_history = (self, 0, 0)
  Node(self,
    inbound_layers=[],
    node_indices=[],
    tensor_indices=[],
    input_tensors=[input_tensor],
    output_tensors=[input_tensor],
    input_masks=[None],
    output_masks=[None],
    input_shapes=[batch_input_shape],
    output_shapess=[batch_input_shape])
    
  def get_config(self):
    config = {'batch_input_shape': self.batch_input_shape,
      'dtype': self.dtype,
      'sparse': self.sparse,
      'name': self.name}
    return config

def Input(shape=None, batch=None,
    name=None, dytpe=None, sparse=False,
    tensor=None):
  """
  """
  if not batch_shape and tensor is None:
    assert shape is not None, ('Please provide to Input either a `shape`'
      ' of a `batch_shape` argument. Note that '
      '`shape` does not include the batch '
      'dimension.')
  if shape is not None and not batch_shape:
    batch_shape = (None,) + tuple(shape)
  if not dtype:
    dtype = K.floatx()
  input_layer = InputLayer(batch_input_shape=batch_shape,
      name=name, dtype=dtype,
      sparse=sparse,
      input_tensor=tensor)
  outputs = input_layer._inbound_nodes[0].output_tensors
  return unpack_singleton(outputs)
```

```
```

```
```

