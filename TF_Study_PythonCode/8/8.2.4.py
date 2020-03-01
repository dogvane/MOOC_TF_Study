import tensorflow as tf
import numpy as np

# import tensorflow.examples.tutorials.mnist.input_data as input_data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784], name="x")
y = tf.placeholder(tf.float32, [None, 10], name="y")

w = tf.Variable(tf.random_normal([784, 10]), name="w")
b = tf.Variable(tf.zeros([10]), name="b")

forward = tf.matmul(x, w) + b    # 前向计算
pred = tf.nn.softmax(forward)

train_epochs = 50
batch_size = 100
total_batch = int(mnist.train.num_examples/batch_size)
display_step = 1
learning_rate = 0.01

# 交叉熵损失函数
loss_function = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# 梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

# 定义准确率
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(train_epochs):
    for batch in range(total_batch):
        xs, ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x:xs, y:ys})
    
    # 完成一个批次后，计算误差与准确率
    loss, acc = sess.run([loss_function, accuracy],
                        feed_dict={x:mnist.validation.images, y:mnist.validation.labels})
    
    if(epoch+1) % display_step == 0:
        print("train epoch:", epoch+1, ' Loss=', loss, " accuracy=", acc)

print("finish!")

