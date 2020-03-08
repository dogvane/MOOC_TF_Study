import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import urllib.request
import os
import tarfile
import pickle as p
import numpy as np

url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
filepath = '../data/cifar-10-python.tar.gz'
dataFilePath = '../data/cifar-10-batches-py'
if not os.path.isfile(filepath):
    result = urllib.request.urlretrieve(url, filepath)
    print('downloaded:', result)
else:
    print('Data file already exists.')

if not os.path.exists(dataFilePath):
    tfile = tarfile.open(filepath, 'r:gz')
    result = tfile.extractall('../data/')
    print('Extracted to ../data/cifar-10-batches-py')
else:
    print('Directory already exits.')


def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        # 样本的的存储格式
        # <1 x label><3072 x pixel> (3072=32x32x3)
        # ...
        # <1 x label><3072 x pixel>

        data_dict = p.load(f, encoding='bytes')
        images = data_dict[b'data']
        labels = data_dict[b'labels']

        # 调整数据的结构为 BCWH
        images = images.reshape(10000, 3, 32, 32)
        # tf处理图像数据的结构是 BWHC
        # 把通道数据C移动到最后一个维度
        images = images.transpose(0, 2, 3, 1)
        labels = np.array(labels)

        return images, labels


def load_CIFAR_data(data_dir):

    images_train = []
    labels_train = []
    for i in range(5):
        f = os.path.join(data_dir, 'data_batch_%d' % (i + 1))
        print('loading', f)
        image_batch, label_batch = load_CIFAR_batch(f)
        images_train.append(image_batch)
        labels_train.append(label_batch)

        del image_batch, label_batch

    xTrain = np.concatenate(images_train)
    yTrain = np.concatenate(labels_train)
    xTest, yTest = load_CIFAR_batch(os.path.join(data_dir, 'test_batch'))

    print('finished loadding CIFAR-10 data')

    return xTrain, yTrain, xTest, yTest


xtrain, ytrain, xtest, ytest = load_CIFAR_data(dataFilePath)

# print('training data shape:', xtrain.shape)
# print('training labels shape:', ytrain.shape)

# print('test data shape:', xtest.shape)
# print('test labels shape:', ytest.shape)


xtrain_normalize = xtrain.astype('float32') / 255.0
xtest_normalize = xtest.astype('float32') / 255.0

encoder = OneHotEncoder(sparse=False)
yy = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
encoder.fit(yy)

ytrain_reshape = ytrain.reshape(-1, 1)
ytrain_onehot = encoder.transform(ytrain_reshape)

ytest_reshape = ytest.reshape(-1, 1)
ytest_onehot = encoder.transform(ytest_reshape)

# print('ytrain_onehot.shape: ', ytrain_onehot.shape)
# print('ytrain[:5]', ytrain[:5])
# print('ytrain_onehot[:5]', ytrain_onehot[:5])


tf.reset_default_graph()

# 定义权值


def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="w")

# 定义偏置


def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), name="b")

# 定义卷积操作


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

# 定义池化操作
# 步长为2， 原尺寸的长度和宽度除以2


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# 输入层
with tf.name_scope("input_layer"):
    x = tf.placeholder('float', shape=[None, 32, 32, 3], name='x')

# 第一个卷积层
# 输入通道：3 输出通道 ：32， 卷积后图像尺寸不变，仍然是32x32
with tf.name_scope("conv_1"):
    w1 = weight([3, 3, 3, 32])
    b1 = bias([32])
    conv_1 = conv2d(x, w1) + b1
    conv_1 = tf.nn.relu(conv_1)

# 第一个池化层
# 将 32x32的图像缩小为16x16, 池化不改变通道数量，仍然是3个
with tf.name_scope("pool_1"):
    pool_1 = max_pool_2x2(conv_1)

# 第2个卷积层
# 输入通道 32 输出通道64，卷积后图像尺寸不变，仍然是 16x16
with tf.name_scope('conv_2'):
    w2 = weight([3, 3, 32, 64])
    b2 = bias([64])
    conv_2 = conv2d(pool_1, w2) + b2
    conv_2 = tf.nn.relu(conv_2)

# 第2个池化层
# 将16x 16的图像缩小为8x8
with tf.name_scope("pool_2"):
    pool_2 = max_pool_2x2(conv_2)

# 全连接层
# 将第二个池化层的64个8x8的图像，转换为一维向量，长度是 64x8x8=4096
with tf.name_scope('fc'):
    w3 = weight([4096, 128])
    b3 = bias([128])
    flat = tf.reshape(pool_2, [-1, 4096])
    h = tf.nn.relu(tf.matmul(flat, w3) + b3)
    h_dropout = tf.nn.dropout(h, keep_prob=0.8)

with tf.name_scope('output_layer'):
    w4 = weight([128, 10])
    b4 = bias([10])
    pred = tf.nn.softmax(tf.matmul(h_dropout, w4) + b4)

with tf.name_scope('optimizer'):
    y = tf.placeholder('float', shape=[None, 10], name='label')
    loss_function = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(
        learning_rate=0.0001).minimize(loss_function)

with tf.name_scope('evaluation'):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

train_epochs = 900
batch_size = 1500
total_batch = int(len(xtrain) / batch_size)
test_batch = int(len(xtest) / batch_size)


# 记录已经训练的次数，不限于运算
epoch = tf.Variable(0, name='epoch', trainable=False)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

ckpt_dir = '../saver/CIFAR10_log'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

saver = tf.train.Saver(max_to_keep=1)

ckpt = tf.train.latest_checkpoint(ckpt_dir)
if ckpt != None:
    saver.restore(sess, ckpt)
else:
    print('Training from scratch.')

# 获取训练的次数
start = sess.run(epoch)
print("Training starts from {} epoch.".format(start + 1))


def get_train_batch(number, batch_size):
    return xtrain_normalize[number*batch_size:(number+1)*batch_size], ytrain_onehot[number*batch_size:(number+1)*batch_size]

def get_test_batch(number, batch_size):
    return xtest_normalize[number*batch_size:(number+1)*batch_size], ytest_onehot[number*batch_size:(number+1)*batch_size]


for ep in range(start, train_epochs):
    for i in range(total_batch):
        batch_x, batch_y = get_train_batch(i, batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if i % 100 == 0:
            print('step {}'.format(i), "finish")

    accuracy_list = []
    loss_list = []

    for i in range(test_batch):
        batch_x, batch_y = get_test_batch(i, batch_size)
        loss, acc = sess.run([loss_function, accuracy],
                            feed_dict={x: batch_x, y: batch_y})
        loss_list.append(loss)
        accuracy_list.append(acc)

    losst = np.array(loss_list);
    acc_t = np.array(accuracy_list);

    print('训练轮次 ', sess.run(epoch) + 1, "loss=", np.sum(losst) / len(loss_list), "accuracy=", np.sum(acc_t) / len(accuracy_list))

    sess.run(epoch.assign(ep + 1))

saver.save(sess, os.path.join(ckpt_dir, "CIFAR10_cnn_model.cpkt"), global_step=train_epochs)
print(" train finished takes.")
