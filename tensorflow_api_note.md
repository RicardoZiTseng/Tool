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

- [tf.name_scope()与tf.variable_sope()的区别](https://www.zhihu.com/question/54513728/answer/181819324)

- [tf.add_to_collection()](https://blog.csdn.net/william_hehe/article/details/78732497)
  - tf.add_to_collection(name,value) 
    > 功能：将变量添加到名为name的集合中去。 
    > 参数：（1）name：集合名（2）value：被添加的变量
  - tf.get_collection(key,scope=None)
    > 功能：获取集合中的变量。
    > 参数：（1）key：集合名
  - tf.add_n(inputs，name=None) 
    > 功能：以元素方式添加所有输入张量。
    > 参数：（1）inputs：张量对象列表，每个对象具有相同的形状和类型。（2）name：操作的名称（可选）。
    
 - [tf辅助工具-tensorflow slim](https://www.cnblogs.com/Libo-Master/p/8466104.html)
    - slim的导入方法
      ```
      import tensorflow as tf
      import tensorflow.contrib.slim as slim
      ```
