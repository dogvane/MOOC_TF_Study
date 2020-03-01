import tensorflow as tf

scalar = tf.constant(100)
vector = tf.constant([1,2,3,4,5])
matrix = tf.constant([[1,2,3], [4,5,6]])
cube_matrix = tf.constant([[[1], [2], [3]],[[4], [5], [6]],[[7],[8],[9]]])

print(scalar.get_shape())
print(vector.get_shape())
print(matrix.get_shape())
print(cube_matrix.get_shape())

tens1 = tf.constant([[[1,2],[2,3]], [[3,4],[5,6]]])

sess = tf.Session()
ret = sess.run(tens1)
print(ret)
print(ret[1,1,0])
sess.close()

# 类型不匹配的例子
a = tf.constant([1,2], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b