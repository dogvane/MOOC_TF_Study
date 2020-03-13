using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using MOOC_TF_Study.Common;
using NumSharp;
using NumSharp.Backends.Unmanaged;
using Tensorflow;
using Tensorflow.Hub;
using static Tensorflow.Binding;

namespace MOOC_TF_Study._10
{
    class _10_4_3
    {
        public static void Run()
        {
            fun1();
        }

        private static void fun1()
        {
            tf.reset_default_graph();

            // 定义权值
            Func<int[], RefVariable> weight = (shape) => tf.Variable(tf.truncated_normal(shape, stddev: 0.1f), dtype: TF_DataType.TF_FLOAT, name: "w");

            // 定义偏置
            Func<TensorShape, Tensor> bias = (shape) => tf.Variable(tf.constant(0.1f, shape: shape), dtype: TF_DataType.TF_FLOAT, name: "b");

            // 定义卷积操作
            Func<Tensor, RefVariable, Tensor> conv2d = (x, w) => tf.nn.conv2d(x, w, strides: new[] { 1, 1, 1, 1 }, padding: "SAME");

            // 定义池化操作
            Func<Tensor, Tensor> max_pool_2x2 = (x) => tf.nn.max_pool(x, ksize: new[] { 1, 2, 2, 1 }, strides: new[] { 1, 2, 2, 1 }, padding: "SAME");

            // 输入层
            var x = tf.placeholder(TF_DataType.TF_FLOAT, shape: new int[] { -1, 32, 32, 3 }, name: "x");

            // 第一个卷积层
            var w1 = weight(new[] { 3, 3, 3, 32 });
            var b1 = bias((32));
            var conv_1 = conv2d(x, w1) + b1;
            conv_1 = tf.nn.relu(conv_1);

            // 第一个池化层
            var pool_1 = max_pool_2x2(conv_1);

            // 第二个卷积层
            var w2 = weight(new[] { 3, 3, 32, 64 });
            var b2 = bias((64));
            var conv_2 = conv2d(pool_1, w2) + b2;
            conv_2 = tf.nn.relu(conv_2);

            // 第二个池化层
            var pool_2 = max_pool_2x2(conv_2);

            // Tensor h_dropout = null;
            // 全连接层
                var w3 = weight(new[] { 4096, 128 });
                var b3 = bias((128));
                var flat = tf.reshape(pool_2, np.array(-1, 4096));
                var h = tf.nn.relu(tf.matmul(flat, w3) + b3);
                var h_dropout = tf.nn.dropout(h, keep_prob: tf.constant(0.8f));

            // 输出层
            var w4 = weight(new[] { 128, 10 });
            var b4 = bias((10));
            var forward = tf.matmul(h_dropout, w4) + b4;
            var pred = tf.nn.softmax(forward);

            // 优化函数
            var y = tf.placeholder(TF_DataType.TF_FLOAT, (-1, 10), "label");
            var loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels:y, logits: forward, name: "loss"));
            var optimizer = tf.train.AdamOptimizer(0.0001f).minimize(loss_function);

            var correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1));
            var accuracy = tf.reduce_mean(tf.cast(correct_prediction, TF_DataType.TF_FLOAT));
        }
    }
}
