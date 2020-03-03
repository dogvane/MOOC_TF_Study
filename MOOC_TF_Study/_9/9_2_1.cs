using System;
using System.Collections.Generic;
using System.Text;
using MOOC_TF_Study.Common;
using NumSharp;
using Tensorflow;
using Tensorflow.Hub;
using static Tensorflow.Binding;

namespace MOOC_TF_Study._9
{
    class _9_2_1
    {
        public static void Run()
        {
            fun1();
        }

        private static void fun1()
        {
            var path = "../../../../data/MNIST_data/";
            // 这里需要引用 Tensorflow.Hub 项目
            var mnist = MnistModelLoader.LoadAsync(path, oneHot:true).Result;

            // 输入层
            var x = tf.placeholder(TF_DataType.TF_DOUBLE, (-1, 784), name: "x");
            var y = tf.placeholder(TF_DataType.TF_DOUBLE, (-1, 10), name: "y");

            // 隐含层
            var H1_NN = 256;
            var H2_NN = 64;

            // 输入层 => 隐含层1
            var w1 = tf.Variable(tf.random_normal((784, H1_NN), dtype:TF_DataType.TF_DOUBLE));
            var b1 = tf.Variable(tf.zeros((H1_NN), dtype: TF_DataType.TF_DOUBLE));

            // 隐含层1 => 隐含层2
            var w2 = tf.Variable(tf.random_normal((H1_NN, H2_NN), dtype: TF_DataType.TF_DOUBLE));
            var b2 = tf.Variable(tf.zeros((H2_NN), dtype: TF_DataType.TF_DOUBLE));

            // 隐含层2 => 输出层
            var w3 = tf.Variable(tf.random_normal((H2_NN, 10), dtype: TF_DataType.TF_DOUBLE));
            var b3 = tf.Variable(tf.zeros((10), dtype: TF_DataType.TF_DOUBLE));

            var y1 = tf.nn.relu(tf.matmul(x, w1) + b1);
            var y2 = tf.nn.relu(tf.matmul(y1, w2) + b2);

            var forward = tf.matmul(y2, w3) + b3;
            var pred = tf.nn.softmax(forward);

            var train_epochs = 50;
            var batch_size = 100;
            var total_batch = (int) mnist.Train.NumOfExamples / batch_size;
            var display_step = 1;
            var learning_rate = 0.01f;

            // var loss_function = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices: 1));
            var loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels: y, logits: forward));
            // var optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function);
            var optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function);

            // 准确率的定义
            var correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1));
            var accuracy = tf.reduce_mean(tf.cast(correct_prediction, TF_DataType.TF_DOUBLE));

            using (var sess = tf.Session())
            {
                var init = tf.global_variables_initializer();
                sess.run(init);

                for (var epoch = 0; epoch < train_epochs; epoch++)
                {
                    for (var batch = 0; batch < total_batch; batch++)
                    {
                        var (xs, ys) = mnist.Train.GetNextBatch(batch_size);
                        sess.run(optimizer, new FeedItem(x, xs), new FeedItem(y, ys));
                    }

                    var (loss, acc) = sess.run((loss_function, accuracy),
                        new FeedItem(x, mnist.Validation.Data),
                        new FeedItem(y, mnist.Validation.Labels));

                    if ((epoch + 1) % display_step == 0)
                        Console.WriteLine($"train epoch:{epoch + 1}   loss={loss} accuracy={acc}");
                }

                var accu_test = sess.run(accuracy, (x, mnist.Test.Data), (y, mnist.Test.Labels));
                Console.WriteLine($"准确率：{accu_test}");
            }
        }
    }
}
