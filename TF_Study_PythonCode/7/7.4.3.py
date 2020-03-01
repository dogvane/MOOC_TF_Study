import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

learning_rate=0.01
train_epochs = 50

df = pd.read_csv("../data/boston.csv", header=0)
df = df.values

for i in range(12):
    df[:, i] = df[:, i]/ (df[:,i].max() - df[:, i].min())    

x_data = df[:,:12]
y_data = df[:,12];

x = tf.placeholder(tf.float32, [None, 12], name="X")
y = tf.placeholder(tf.float32, [None, 1], name="Y")

with tf.name_scope("Model"):
    w = tf.Variable(tf.random_normal([12, 1], stddev=0.01), name="W")

    b = tf.Variable(1.0, name="b")

    def model(x, w, b):
        return tf.matmul(x, w) + b

    pred = model(x, w, b)
    
with tf.name_scope("LossFunction"):
    loss_function = tf.reduce_mean(tf.pow(y - pred, 2))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)
loss_list = []

for epoch in range(train_epochs):
    loss_sum = 0.0
    for xs, ys in zip(x_data, y_data):
        xs = xs.reshape(1, 12)
        ys = ys.reshape(1,1)
        _, loss = sess.run([optimizer, loss_function], feed_dict={x: xs, y:ys})

        loss_sum = loss_sum + loss
    
    xvalues, yvalues = shuffle(x_data, y_data)
    
    b0temp = b.eval(session=sess)
    w0temp = w.eval(session=sess)

    loss_average=loss_sum/len(y_data)
    loss_list.append(loss_average)

    print("epoch=", epoch+1," loss=", loss_average, " b=", b0temp, " w=", w0temp)
    

plt.plot(loss_list)
plt.show()