from time import time
import tensorflow as tf
import numpy as np

# import tensorflow.examples.tutorials.mnist.input_data as input_data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

# 输入层
x = tf.placeholder(tf.float32, [None, 784], name="x")
y = tf.placeholder(tf.float32, [None, 10], name="y")

# 隐藏层（隐含层）
H1_NN = 256 # 隐藏层1
H2_NN = 64  # 隐藏层2

# 输入层 --> 隐藏层1
W1 = tf.Variable(tf.truncated_normal([784, H1_NN], stddev=0.1))
b1 = tf.Variable(tf.zeros([H1_NN]))

# 隐藏层1 --> 隐藏层2
W2 = tf.Variable(tf.truncated_normal([H1_NN, H2_NN], stddev=0.1))
b2 = tf.Variable(tf.zeros([H2_NN]))

# 隐藏层2 --> 输出层
W3 = tf.Variable(tf.truncated_normal([H2_NN, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

Y1 = tf.nn.relu(tf.matmul(x, W1) + b1)  # 隐藏层的激活函数
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + b2)

forward = tf.matmul(Y2, W3) + b3
pred = tf.nn.softmax(forward)

# 训练参数
train_epochs = 40
batch_size = 50
total_batch = int(mnist.train.num_examples/batch_size)
display_step = 1
learning_rate = 0.01

# 交叉熵损失函数
loss_function = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=forward, labels=y))

# Adam 优化器
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

# 定义准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 初始化开始训练的时间
startTime = time()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(train_epochs):
    for batch in range(total_batch):
        xs, ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: xs, y: ys})

    # 完成一个批次后，计算误差与准确率
    loss, acc = sess.run([loss_function, accuracy],
                         feed_dict={x: mnist.validation.images,
                                    y: mnist.validation.labels})

    if(epoch+1) % display_step == 0:
        print("train epoch:", epoch+1, ' Loss=', loss, " accuracy=", acc)

# 评估模型
accu_test = sess.run(accuracy, feed_dict={
                     x: mnist.test.images, y: mnist.test.labels})
print("准确率：", accu_test)

prediction_result = sess.run(
    tf.argmax(pred, 1), feed_dict={x: mnist.test.images})

print('预测结果：', prediction_result[0:10])

compare_lists = prediction_result == np.argmax(mnist.test.labels, 1)
print('预测与实际的结果比较：', compare_lists)

err_lists = [i for i in range(len(compare_lists)) if compare_lists[i] == False]
print('错误数量：', len(err_lists), "   错误内容：", err_lists)

