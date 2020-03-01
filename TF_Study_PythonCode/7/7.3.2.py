import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 12], name="X")
y = tf.placeholder(tf.float32, [None, 1], name="Y")

with tf.name_scope("Model"):
    w = tf.Variable(tf.random_normal([12, 1], stddev=0.01), name="W")

    b = tf.Variable(1.0, name="b")

    def model(x, w, b):
        return tf.matmul(x, w) + b

    pred = model(x, w, b)
