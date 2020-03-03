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
H1_NN = 256
W1 = tf.Variable(tf.random_normal([784, H1_NN]))
b1 = tf.Variable(tf.zeros([H1_NN]))

Y1 = tf.nn.relu(tf.matmul(x, W1) + b1)  # 隐藏层的激活函数

# 输出层
W2 = tf.Variable(tf.random_normal([H1_NN, 10]))
b2 = tf.Variable(tf.zeros([10]))

forward = tf.matmul(Y1, W2) + b2
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


def print_predict_errs(labels, prediction):
    count = 0
    compare_lists = (prediction == np.argmax(labels, 1))
    err_lists = [i for i in range(
        len(compare_lists)) if compare_lists[i] == False]
    for x in err_lists:
        print("index=", str(x), "标签值=", np.argmax(
            labels[x]), " 预测值=", prediction[x])
        count = count+1
    print("总计：", str(count))

print_predict_errs(labels=mnist.test.labels, prediction=prediction_result)

# 以下为扩展方法

import matplotlib.pyplot as plt

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

err_images=[]
err_labels=[]
err_prediction=[]

for x in err_lists:
    err_images.append(mnist.test.images[x])
    err_labels.append(mnist.test.labels[x])
    err_prediction.append(prediction_result[x])

plot_images_labels_prediction(err_images, err_labels, err_prediction, 0, 25)
