# coding: utf-8
# ---
# @File: test2_ragged.py
# @Time: 2020/8/14 15:10
# @Author: Xin Zhang
# @E-mail: meetdevin.zh@outlook.com
# @desc: 
# ---
import numpy as np
import tensorflow as tf

inputs = np.random.random(size=[2, 2, 4, 4])

c = tf.ones(
    [2, 2, 2, 4],
    dtype=tf.double,
    name=None
)



# line = np.random.randint(10, size=[2, 4])


class MyModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten = tf.keras.layers.Flatten(data_format='channels_first')
        self.get_line = tf.keras.layers.Dense(units=4, activation=tf.nn.sigmoid, use_bias=False,
                                              kernel_initializer='glorot_uniform')

    def to_bool(self, x):
        median = np.median(x)
        return tf.less(median, x)

    def __call__(self, inputs, *args, **kwargs):
        line = self.get_line(self.flatten(inputs))

        mask_line = tf.map_fn(fn=self.to_bool, elems=line, dtype=bool)  # [?, units]
        # [?, h]

        mask = tf.expand_dims(mask_line, axis=1)
        mask = tf.tile(mask, [1, 2, 1])
        mask_inverse = tf.equal(mask, False)  # 取反

        feat_attention = tf.ragged.boolean_mask(data=inputs, mask=mask)  # 维度变化，导致梯度消失
        feat_dropout = tf.ragged.boolean_mask(data=inputs, mask=mask_inverse)
        feat_attention = feat_attention.to_tensor()
        feat_dropout = feat_dropout.to_tensor()

        out = tf.add(feat_attention, tf.multiply(feat_dropout, 0.0001))

        print('a', inputs)
        print('mask', mask)
        print('mask_inverse', mask_inverse)
        print('feat_attention', out)

        return out


model = MyModel()
feat_attention = model(inputs)


optimizer = tf.optimizers.Adam(0.1)
with tf.GradientTape() as tape:
    loss = tf.reduce_mean(tf.math.subtract(feat_attention, c))
grads = tape.gradient(loss, model.trainable_weights)
optimizer.apply_gradients(zip(grads, model.trainable_weights))
