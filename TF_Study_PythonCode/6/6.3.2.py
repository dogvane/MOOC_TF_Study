import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def model(x, w, b):
    return tf.multiply(x, w) + b

# np.random.seed(5)


# 等差数列，100个点， -1~+1之间
x_data = np.linspace(-1, 1, 100)

y_data = 2 * x_data + 1.0 + np.random.randn(*x_data.shape)*0.4
# print(y_data)
y_data = [-0.77710136, -0.54350417, -0.86547821, -0.55992125, -0.837342, -1.33467084,
          -1.2444469, -0.82076245, -1.32381571, -0.34003532, -
          0.32571077, -0.20681164, -0.5856616, -0.29864504,
          -0.93447063, -0.14173247, -0.32288982, 0.1057742, -
          0.61476565, -0.25148246, 0.22452511, 0.16641991,
          0.17846703, -0.01658544, -0.01971127, -
          0.09563799, 0.61697675, -0.26326721, 0.25372583, 0.79789147,
          0.35918527, 0.39203612, -0.41106826, 0.56028576, 1.02617934, -
          0.10681189, 0.88786259, 1.00214365,
          0.47857213, 0.52466823, 0.48748845, 0.78675198, 1.44487602, 1.07785516, 0.78161655, 0.81507455,
          0.79752379, 1.07105367, 1.21923287, 1.14126001, 1.19698625, 0.77071674, 1.32659001, 1.1879609,
          1.51570375, 1.32938085, 1.04449763, 1.19243454, 2.03804625, 1.22214014, 1.36984456, 1.60431261,
          1.34007491, 1.30594774, 1.31724576, 1.86786032, 0.96547259, 1.65217273, 1.68022047, 1.13190774,
          2.34050031, 2.5374313, 1.98572393, 1.9033125, 1.91291599, 1.65070139, 1.85006016, 2.16014724,
          2.30318022, 1.56625583, 2.26075892, 2.01487454, 2.59661301, 2.50489875, 2.54016046, 2.43119098,
          2.02573928, 2.13201859, 2.33623011, 2.26637834, 3.25502429, 2.92800369, 2.22934641, 2.79055945,
          3.67187637, 2.87554269, 3.46687067, 2.72311618, 3.65135512, 2.95524828]

x = tf.placeholder("float", name="x")
y = tf.placeholder("float", name="y")

w = tf.Variable(1.0, name="w0")
b = tf.Variable(0.0, name="b0")

pred = model(x, w, b)

train_epochs = 10
learning_rate = 0.05

loss_function = tf.reduce_mean(tf.square(y - pred))
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(loss_function)

sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)

for epoch in range(train_epochs):
    for xs, ys in zip(x_data, y_data):
        _, loss = sess.run([optimizer, loss_function], feed_dict={x: xs, y: ys})

x_test = 3.21

predict = sess.run(pred, feed_dict={x:x_test})
print("预测值：%f" % predict)

target = 2 * x_test + 1.0
print("目标值：%f" % target)