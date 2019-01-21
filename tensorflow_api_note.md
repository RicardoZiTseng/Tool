- tf.split

  ```
  split(
    value,
    num_or_size_splits,
    axis=0,
    num=None,
    name='split'
  )
  ```
  - value： 输入的tensor 
  - num_or_size_splits: 如果是个整数n，就将输入的tensor分为n个子tensor。如果是个tensor T，就将输入的tensor分为len(T)个子tensor。 
  - axis： 默认为0，计算value.shape[axis], 一定要能被num_or_size_splits整除。 
  
  **example**
  ```
  >>> import numpy as np
  >>> import tensorflow as tf
  >>> s=np.random.randn(4,10,10,3)
  >>> r,g,b=tf.split(s,3,3)
  >>> print(r.shape, g.shape, b.shape)
  (4, 10, 10, 1) (4, 10, 10, 1) (4, 10, 10, 1)
  ```
  
  **reference**
  - [TensorFlow笔记——tf.split()拆分tensor和tf.squeeze()](https://blog.csdn.net/liuweiyuxiang/article/details/81192547)
