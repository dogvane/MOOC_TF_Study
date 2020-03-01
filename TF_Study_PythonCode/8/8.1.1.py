import tensorflow as tf
import numpy as np

# import tensorflow.examples.tutorials.mnist.input_data as input_data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

print('训练集 train 数量：', mnist.train.num_examples,
    ', 验证集 validation 数量：', mnist.validation.num_examples,
    ', 测试集 test 数量：', mnist.test.num_examples)

print('train images shape: ', mnist.train.images.shape, 
        'labels shape: ', mnist.train.labels.shape)

image0 = mnist.train.images[0]

print('imageLen= ', len(image0))
print('image.shape =', image0.shape)
print('image data=', image0)
print('image.reshape =', image0.reshape(28,28))

import matplotlib.pyplot as plt
def plot_image(image):
    plt.imshow(image.reshape(28,28), cmap='binary')
    plt.show()

plot_image(mnist.train.images[1])
