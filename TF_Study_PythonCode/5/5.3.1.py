import tensorflow as tf

logdir = 'r:/log'

input1 = tf.constant([1.0, 2.0, 3.0], name="input")
input2 = tf.Variable(tf.random_uniform([3]), name="input2")

output = tf.add_n([input1, input2], name="add")
writer = tf.compat.v1.summary.FileWriter(logdir, tf.compat.v1.get_default_graph())
writer.close();
