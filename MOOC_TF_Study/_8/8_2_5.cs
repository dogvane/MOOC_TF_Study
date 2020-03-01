using System;
using System.Collections.Generic;
using System.Text;
using MOOC_TF_Study.Common;
using NumSharp;
using Tensorflow;
using Tensorflow.Hub;
using static Tensorflow.Binding;

namespace MOOC_TF_Study._8
{
    class _8_2_5
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

            var x = tf.placeholder(TF_DataType.TF_DOUBLE, (-1, 784), name: "x");
            var y = tf.placeholder(TF_DataType.TF_DOUBLE, (-1, 10), name: "y");

            var w = tf.Variable(tf.random_normal((784, 10), dtype:TF_DataType.TF_DOUBLE), dtype: TF_DataType.TF_DOUBLE, name: "x");
            var b = tf.Variable(tf.zeros(10, dtype:TF_DataType.TF_DOUBLE), dtype:TF_DataType.TF_DOUBLE, name: "b");

            var forward = tf.matmul(x, w) + b;
            var pred = tf.nn.softmax(forward);

            var train_epochs = 50;
            var batch_size = 100;
            var total_batch = (int) mnist.Train.NumOfExamples / batch_size;
            var display_step = 1;
            var learning_rate = 0.01f;

            // var loss_function = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices: 1));
            var loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels: y, logits: forward));
            var optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function);

            var correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1));
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

                var prediction_result = sess.run(tf.argmax(pred, 1), new FeedItem(x, mnist.Test.Data));
                Console.WriteLine($"验证集结果：{prediction_result["0:10"]}");

                // 因为，这里没法画图，所以，只做一个简单的文字输出了
                int num = 10;
                int startIndex = 0;
                var labels = mnist.Test.Labels;

                for (var i = startIndex; i < num; i++)
                {
                    Console.WriteLine($"标签值：{np.argmax(labels[i])}  预测值：{prediction_result[i]}");
                }
            }
        }
    }
}
