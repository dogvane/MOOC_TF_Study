import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
# loss_function = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# 带 softmax 的交叉熵函数
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=forward))

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

prediction_result = sess.run(tf.argmax(pred, 1), feed_dict={x: mnist.test.images})
print('prediction_result: ', prediction_result[0:10])


def plot_images_labels_prediction(images,
                                  labels,
                                  prediction,
                                  index,
                                  num=10):
    fig = plt.gcf()
    fig.set_size_inches(10, 12)
    if num > 25:
        num = 25    # 最多显示25个子图
    for i in range(0, num):
        ax = plt.subplot(5, 5, i+1) # 获取要处理图像的位置
        ax.imshow(np.reshape(images[index], (28, 28)), cmap='binary') # 显示第index个图像
        title = "label=" + str(np.argmax(labels[index]))
        if len(prediction) > 0:
            title += ", predict=" + str(prediction[index])
        
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])   # 不显示坐标轴
        ax.set_yticks([])
        index += 1
    
    plt.show()

plot_images_labels_prediction(mnist.test.images, mnist.test.labels, prediction_result, 10, 25)
