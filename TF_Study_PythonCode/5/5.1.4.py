import tensorflow as tf

# tf 1.5 的 reset_default_graph() 方法，和原教材的版本不一致了
tf.compat.v1.reset_default_graph()

a = tf.Variable( 1, name="a")
b = tf.add(a, 1, name="b")
c = tf.multiply(b, 4, name="c")
d = tf.subtract(c, b, name="d")

logdir = 'r:/log'
writer = tf.compat.v1.summary.FileWriter(logdir, tf.compat.v1.get_default_graph())
writer.close()
