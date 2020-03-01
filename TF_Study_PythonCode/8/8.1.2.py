import tensorflow as tf
import numpy as np

# import tensorflow.examples.tutorials.mnist.input_data as input_data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

print(mnist.train.labels[1])
print(np.argmax(mnist.train.labels[1])) # 最大数的索引

mnist_no_one_hot = input_data.read_data_sets("../data/MNIST_data/", one_hot=False)

print(mnist_no_one_hot.train.labels[0:10])
