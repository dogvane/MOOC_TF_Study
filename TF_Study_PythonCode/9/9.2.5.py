import tensorflow as tf

# import tensorflow.examples.tutorials.mnist.input_data as input_data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

# 输入层
x = tf.placeholder(tf.float32, [None, 784], name="x")
y = tf.placeholder(tf.float32, [None, 10], name="y")

def fcn_layer(inputs,
            input_dim,
            output_dim,
            activation=None):
    W = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
    b = tf.Variable(tf.zeros([output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    
    return outputs

h1 = fcn_layer(inputs=x, input_dim=784, output_dim=256, activation=tf.nn.relu)
forward = fcn_layer(inputs=h1, input_dim=256, output_dim=10, activation=None)
pred = tf.nn.softmax(forward)


# 隐藏层（隐含层）
H1_NN = 256 # 隐藏层1
H2_NN = 64  # 隐藏层2
H3_NN = 32  # 隐藏层3

h1 = fcn_layer(inputs=x, input_dim=784, output_dim=H1_NN, activation=tf.nn.relu)
h2 = fcn_layer(inputs=h1, input_dim=H1_NN, output_dim=H2_NN, activation=tf.nn.relu)
h3 = fcn_layer(inputs=h2, input_dim=H2_NN, output_dim=H3_NN, activation=tf.nn.relu)

forward = fcn_layer(inputs=h3, input_dim=H3_NN, output_dim=10, activation=None)
pred = tf.nn.softmax(forward)

# 训练参数
train_epochs = 40
batch_size = 50
total_batch = int(mnist.train.num_examples/batch_size)
display_step = 1
learning_rate = 0.01
save_step=5

# 检查保存的目录
import os
ckpt_dir = "../saver"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# 交叉熵损失函数
loss_function = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=forward, labels=y))

# Adam 优化器
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

# 定义准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

ckpt = tf.train.get_checkpoint_state(ckpt_dir)

if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('模型已恢复：', ckpt.model_checkpoint_path)

# 评估模型
accu_test = sess.run(accuracy, feed_dict={
                     x: mnist.test.images, y: mnist.test.labels})
print("准确率：", accu_test)
