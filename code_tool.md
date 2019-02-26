#### 一些有用的代码片段

1. 数据的随机batch输出：
```python
import numpy as np

class BrainData():
    def __init__(self, data, label, batch_size=64):
        """
            get the batch data in random way
        Args:
            data: tensor of shape [size, width, height]
            label: tensor of shape [size, width, height, class_num]
            batch_size: number of image batches
        """
        self.data = data[400:]
        self.label = label[400:]
        self.test_data = data[:400]
        self.test_label = label[:400]
        self.width = self.data.shape[1]
        self.height = self.data.shape[2]
        self.batch_size = batch_size
        self.batch_num = self.data.shape[0] // self.batch_size
        self.cursor = 0
        self.random_index = np.arange(0, self.data.shape[0])
        self.epoch = 1
        np.random.shuffle(self.random_index)

    def get_batch(self):
        if self.cursor >= self.batch_num:
            self.cursor = 0
            self.epoch += 1
            np.random.shuffle(self.random_index)

        batch_data, batch_label = self._next_batch()
        self.cursor += 1
        return batch_data, batch_label

    def _next_batch(self):
        images = np.zeros((self.batch_size, self.width, self.height))
        labels = np.zeros((self.batch_size, self.width, self.height, self.label.shape[-1]))

        count = 0

        while count < self.batch_size:
            images[count, :, :] = self.data[self.random_index[self.cursor*self.batch_size + count]]
            labels[count, :, :, :] = self.label[self.random_index[self.cursor*self.batch_size + count]]
            count += 1

        return images, labels
```

2. 语义分割中image label的制作
```python
import numpy as np

def labels_convert(mask):
    """
        Convert the masks to labels.
    Args:
        masks:  int32 - [width, height], a binary matrix of one slice
    Returns:
        masks_label: int 32 - [width, height, 2], binary coding method for mask slice
    """
    width, height = mask.shape
    mask_label = np.zeros((width, height, 2), dtype=np.int8)
    mask_label[...,0] = 1 - mask
    mask_label[...,1] = mask
    return mask_label
```

3. dice系数的计算
```python
def dice_coef(pred_up, labels):
    """
        Params:
        pred_up: numpy array with shape of [?, width, height]
        labels: numpy array with shape of [?, width, height, 2]
    """
    labels = np.array(np.argmax(labels, axis=3), dtype=np.int8)
    # print(labels.shape)
    pred_up = np.array(pred_up, dtype=np.int8)
    # print(pred_up.shape)
    inter = pred_up * labels
    sum_inter = np.sum(np.sum(inter, 1), 1)
    sum_pred_up = np.sum(np.sum(pred_up, 1), 1)
    sum_labels = np.sum(np.sum(labels, 1), 1)
    return np.mean(2*sum_inter/(sum_pred_up + sum_labels))
```

4. Timer工具类，统计时间
```python
import time
import datetime

class Timer(object):
    '''
    A simple timer.
    '''

    def __init__(self):
        self.init_time = time.time()
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.remain_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def remain(self, iters, max_iters):
        if iters == 0:
            self.remain_time = 0
        else:
            self.remain_time = (time.time() - self.init_time) * \
                (max_iters - iters) / iters
        return str(datetime.timedelta(seconds=int(self.remain_time)))
```

5. 双线性插值的deconv核
```python
    def get_deconv_filter(self, f_shape):
        width = f_shape[0]
        height = f_shape[1]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="up_filter", initializer=init,
                              shape=weights.shape)
        return var
```

6. 上采样层
```python
def _upscore_layer(self, bottom, shape, name, wd=0.0005, ksize=4, stride=2):
        """
        Args:
          bottom: shape of [None, width, height, channel]
          shape: the shape of layer which is to be upscored
          wd: weight decay params
        """
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            in_features = bottom.get_shape()[3].value
            new_shape = [shape[0], shape[1], shape[2], self.num_classes]
            out_shape = tf.stack(new_shape)

            f_shape = [ksize, ksize, self.num_classes, in_features]

            num_input = ksize * ksize * in_features * self.num_classes
            stddev = (2/num_input)**0.5

            weights = self.get_deconv_filter(f_shape)

            collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
            weight_decay = tf.multiply(
                tf.nn.l2_loss(weights), wd, name='weight_loss')
            tf.add_to_collection(collection_name, weight_decay)

            deconv = tf.nn.conv2d_transpose(bottom, weights, out_shape, strides=strides, padding='VALID')

        return deconv
```
