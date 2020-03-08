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
        print('images.reshape after', images.shape);

        # tf处理图像数据的结构是 BWHC
        # 把通道数据C移动到最后一个维度
        images = images.transpose(0, 2, 3, 1)

        print('images.transpose after', images.shape);
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

print('training data shape:', xtrain.shape)
print('training labels shape:', ytrain.shape)

print('test data shape:', xtest.shape)
print('test labels shape:', ytest.shape)


# 我要画图了


label_dict = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}


def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    if (num > 10):
        num = 10
    for i in range(0, num):
        ax = plt.subplot(2, 5, i + 1)
        ax.imshow(images[idx], cmap='binary')

        title = str(i) + ',' + label_dict[labels[idx]]
        if len(prediction) > 0:
            title += '=>' + label_dict[prediction[idx]]

        ax.set_title(title, fontsize=10)

        idx += 1
    plt.show()


plot_images_labels_prediction(xtest, ytest, [], 0, 10)
