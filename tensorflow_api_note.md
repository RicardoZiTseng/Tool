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
      - slim定义模型
      - slim中定义一个变量：
      ```python
      # Model Variables
      weights = slim.model_variable('weights', 
                                    shape=[10,10,3,3],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1), 
                                    regularizer=slim.l2_regularizer(0.5),
                                    device='/CPU:0')
      model_variables = slim.get_model_variables()

      # Regular variables
      my_var = slim.variables('my_var',
                              shape=[20,1],
                              initializer=tf.zeros_initializer())
      regular_variables_and_model_variables = slim.get_variables()
      ```
      如上，变量分为两类：模型变量和局部变量。局部变量是不作为模型参数保存的，而模型变量会再save的时候保存下来。这个玩过tensorflow的人都会明白，诸如global_step之类的就是局部变量。slim中可以写明变量存放的设备，正则和初始化规则。

      - slim中实现一个层
        - tensorflow中实现层的方式：
          ```python
          input = ...
          with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal_initializer([3,3,64,128], 
                                                                  dtype=tf.float32,
                                                                  stddev=1e-1),
                                                                  name='weights')
            conv = tf.nn.conv2d(input, kernel, [1,1,1,1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
            trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope)
          ```
        - slim中实现层的方式：
          ```python
          input = ...
          net = slim.conv2d(input, 128, [3,3], scope='conv1_1')
          ```
          如果要叠加多层，可以这样写：
          ```python
          net = ...
          net = slim.conv2d(net, 256, [3,3], scope='conv3_1')
          net = slim.conv2d(net, 256, [3,3], scope='conv3_2')
          net = slim.conv2d(net, 256, [3,3], scope='conv3_3')
          net = slim.max_pool2d(net, [2,2], scope='pool3')
          ```
          但是这样写还是很冗余，所以这里要用到slim中的**repeat**操作：
          ```python
          net = slim.repeat(net, 3, slim.conv2d, 256, [3,3], scope='conv3')
          net = slim.max_pool2d(net, [2,2], scope='pool3')
          ```
          repeat操作处理的是卷积核或输出相同的情况，但是stack是处理卷积核或者输出不一样的情况：
          假设定义三层FC：
          ```python
          x = slim.fully_connected(x, 32, scope='fc/fc_1')
          x = slim.fully_connected(x, 64, scope='fc/fc_2')
          x = slim.fully_connected(x, 128, scope='fc/fc_3')
          ```
          使用stack操作：
          ```python
          x = slim.stack(x, slim.fully_connected, [32, 64, 128], scope='fc')
          ```
          对于卷积层来说，可以这样写：
          ```python
          x = slim.conv2d(x, 32, [3, 3], scope='core/core_1')
          x = slim.conv2d(x, 32, [1, 1], scope='core/core_2')
          x = slim.conv2d(x, 64, [3, 3], scope='core/core_3')
          x = slim.conv2d(x, 64, [1, 1], scope='core/core_4')
          ```
          使用stack操作：
          ```python
          x = slim.stack(x, slim.conv2d, [(32, [3, 3]), 
                                          (32, [1, 1]), 
                                          (64, [3, 3]),
                                          (64, [1, 1])],
                                          scope='core')
          ```
          如果网络中有大量相同的参数，比如：
          ```python
          net = slim.conv2d(inputs, 64, [11, 11], padding='SAME',
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005), scope='conv1')
          net = slim.conv2d(net, 128, [11, 11], padding='VALID',
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005), scope='conv2')
          net = slim.conv2d(net, 256, [11, 11], padding='SAME',
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005), scope='conv3')
          ```
          这里使用**arg_scope**处理一下：
          ```python
          with slim.arg_scope([slim.conv2d], padding='SAME',
                              weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                              weights_regularizer=slim.l2_regularizer(0.005)):
                              net = slim.conv2d(inputs, 64, [11, 11], scope='conv1')
                              net = slim.conv2d(net, 128, [11, 11], padding='VALID', scope='conv2')
                              net = slim.conv2d(net, 256, [11, 11], scope='conv3')
          ```
          arg_scope的作用范围内，定义了指定层的默认参数，如果想特别指定某些层的参数，可以重新赋值（即重写）\
          这里同时使用**arg_scope**和**stack**：
          ```python
          with slim.arg_scope([slim.conv2d], padding='SAME',
                              weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                              weights_regularizer=slim.l2_regularizer(0.005)):
                              net = slim.stack(inputs, slim.conv2d, [ (64,  [11, 11]),
                                                                      (128, [11, 11]),
                                                                      (256, [11, 11])],
                                                                      scope='conv')
          ```
          如果定义的层中还有其他类型的层，则如下定义：
          ```python
          with slim.arg_scope([slim.conv2d, slim.fully_connected],
                              activation_fn=tf.nn.relu, 
                              weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                              weights_regularizer=slim.l2_regularizer(0.0005)):
                              with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
                                net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
                                net = slim.conv2d(net, 256, [5, 5],
                                                  weights_initializer=tf.truncated_normal_initializer(0.03),
                                                  scope='conv2')
                                net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc')
          ```
          VGG网络的定义：
          ```python
          def vgg16(inputs):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                  net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                  net = slim.max_pool2d(net, [2, 2], scope='pool1')
                  net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                  net = slim.max_pool2d(net, [2, 2], scope='pool2')
                  net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                  net = slim.max_pool2d(net, [2, 2], scope='pool3')
                  net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                  net = slim.max_pool2d(net, [2, 2], scope='pool4')
                  net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                  net = slim.max_pool2d(net, [2, 2], scope='pool5')
                  net = slim.fully_connected(net, 4096, scope='fc6')
                  net = slim.dropout(net, 0.5, scope='dropout6')
                  net = slim.fully_connected(net, 4096, scope='fc7')
                  net = slim.dropout(net, 0.5, scope='dropout7')
                  net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
            return net
          ```
- tf.Print() 在tensorflow中用于debug 并观察数据流动
  ```python
  tf.Print(
    input_,
    data,
    message=None,
    first_n=None,
    summarize=None,
    name=None
  )
  ```
    使用 TensorFlow 时，一定要记得所有的东西最终都体现为图的计算。也就是说如果你使用 Python 的 print 命令来输出 TensorFlow 的某个操作，得到的结果仅仅是这个操作的相关描述，因为当前还没有值被传入这个操作。print 命令还常常会显示该节点所期望且已知的维度。
    ```python
    import tensorflow as tf

    a = tf.constant([[4,4],[3,3]])
    print(a)
    ```
    ```
    >>> Tensor("Const:0", shape=(2, 2), dtype=int32)
    ```
    由于上述代码只是描述了一个计算图，直接用print打印输出只会得到数据流图的相关描述，所以想要知道执行计算图后某个具体节点的值，则需要 tf.Print 函数。

    - 调整输出节点的结构位置 \
      在tensorflow的逻辑中，只有需要被执行的图节点才会计算其输出值，所以如果输出语句悬挂于节点之外，则根本不会输出，如下图所示：
      ![](https://pic1.zhimg.com/80/v2-4f3fbe651973b2cb1a8d6465ec6323ac_hd.jpg)
      解决这个问题的办法就是将输出语句嵌入到原来的图当中，就像以下的图一样：
      ![](https://pic1.zhimg.com/80/v2-40073e16a9779df68029217848b3b640_hd.jpg)
      ```python
      node1 = tf.add(input1, input2)
      node1 = tf.Print(node1, [node1], message='something you want to log')
      output = tf.multiply(node1, input3)
      ```
    
    - 一些需要注意的事 \
      注意，在使用 Jupyter notebook 时，输出内容在其 console 的 stderr 中，不是在 notebook 每个代码格子的执行输出中。一定要注意这一点，否则会找不到输出的内容。

- tensorflow中的转置卷积操作
  ```python
    tf.nn.conv2d_transpose(
        value,
        filter,
        output_shape,
        strides,
        padding='SAME',
        data_format='NHWC',
        name=None
    )
  ```
  转置卷积有两个需要注意的点：
  - filter的维度大小为：[filter_size, filter_size, out_channel, in_channel]
  普通的卷积核的维度为：[filter_size, filter_size, in_channel, out_channel]
  - 在tf.nn.conv2d_transpose函数中，参数 output_shape 并不能控制实际输出大小，而是用作检测用的，如果输出节点维度大小与output_shape不符，则会报错


- Tensorflow 中的正则化
  1. 创建一个正则化方法
  2. 将这个正则化方法应用到变量上 \

  对于tf.get_variable()变量初始化方法：
  ```python
    tf.get_variable(
      name,
      shape=None,
      dtype=None,
      initializer=None,
      regularizer=None,
      ...
    )
  ```
  -> 首先创建一个正则化方法
  ```python
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1) #scale代表正则化系数的值
  ```
  -> 然后将正则化方法传给一个variable
  ```python
    a = tf.get_variable(name='I_am_a', regularizer=regularizer, initializer=...)
  ```
  这样就完成了正则化步骤 \

  当然也可以这样写 \
  -> 创建一个variable
  ```python
    var = tf.get_variable(name='var', initializer=...)
  ```
  -> 然后创建正则项
  ```python
    weight_loss = tf.multiply(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
  ```
  -> 然后将正则项加入到tf.GraphKeys.REGULARIZATION_LOSSES集合中
  ```python
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_loss)
  ```
  -> 最后将正则项加入到总的loss中
  ```python
    loss = ... + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  ```
  > 其中, tf.add_n()负责将得到的集合中的所有元素加起来, tf.get_collection()则负责获取指定集合中的所有元素
  
  - Tensorboard的使用
