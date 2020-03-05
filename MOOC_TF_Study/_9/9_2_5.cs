using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using MOOC_TF_Study.Common;
using NumSharp;
using Tensorflow;
using Tensorflow.Hub;
using static Tensorflow.Binding;

namespace MOOC_TF_Study._9
{
    class _9_2_5
    {
        public static void Run()
        {
            fun1();
        }

        static Tensor fcn_layer(Tensor inputs, int input_dim, int output_dim, Func<Tensor, string, Tensor> activation, string activationName=null)
        {
            var w = tf.Variable(tf.truncated_normal((input_dim, output_dim), stddev: 0.1f, dtype:TF_DataType.TF_DOUBLE));
            var b = tf.Variable(tf.zeros((output_dim), TF_DataType.TF_DOUBLE));

            var xwb = tf.matmul(inputs, w) + b;
            if (activation == null)
                return xwb;

            return activation(xwb, activationName);
        }

        private static void fun1()
        {
            var path = "../../../../data/MNIST_data/";
            // 这里需要引用 Tensorflow.Hub 项目
            var mnist = MnistModelLoader.LoadAsync(path, oneHot:true).Result;

            // var savePath = "../../../../netSaver";
            var savePath = @"..\..\..\..\netSaver";

            // var savePath = @"S:\MOOC_TF_Study\netSaver";
            if (!Directory.Exists(savePath))
                Directory.CreateDirectory(savePath);

            // 输入层
            var x = tf.placeholder(TF_DataType.TF_DOUBLE, (-1, 784), name: "x");
            var y = tf.placeholder(TF_DataType.TF_DOUBLE, (-1, 10), name: "y");

            // 隐含层
            var H1_NN = 256;
            var H2_NN = 64;
            var H3_NN = 32;

            var h1 = fcn_layer(inputs: x, input_dim: 784, output_dim: H1_NN, activation: tf.nn.relu);
            var h2 = fcn_layer(h1, H1_NN, H2_NN, tf.nn.relu);
            var h3 = fcn_layer(h2, H2_NN, H3_NN, tf.nn.relu);

            var forward = fcn_layer(h3, H3_NN, 10, null);
            var pred = tf.nn.softmax(forward);

            var train_epochs = 10;
            var batch_size = 100;
            var total_batch = (int) mnist.Train.NumOfExamples / batch_size;
            var display_step = 1;
            var save_step = 10;
            var learning_rate = 0.01f;

            var loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels: y, logits: forward));
            var optimizer = tf.train.AdamOptimizer(learning_rate, dtype: TF_DataType.TF_DOUBLE).minimize(loss_function);

            // 准确率的定义
            var correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1));
            var accuracy = tf.reduce_mean(tf.cast(correct_prediction, TF_DataType.TF_DOUBLE));

            using (var sess = tf.Session())
            {
                var init = tf.global_variables_initializer();
                sess.run(init);
                var saver = tf.train.Saver();

                //var ckpt = tf.train.get_checkpoint_state(savePath);
                //if(ckpt != null && !string.IsNullOrWhiteSpace(ckpt.ModelCheckpointPath))
                //{
                //    saver.restore(sess, ckpt.ModelCheckpointPath);
                //    Console.WriteLine("模型已恢复");
                //}

                var lastFile = tf.train.latest_checkpoint(savePath);
                Console.WriteLine(lastFile);

                var accu_test = sess.run(accuracy, (x, mnist.Test.Data), (y, mnist.Test.Labels));
                Console.WriteLine($"准确率：{accu_test}");
            }
        }
    }
}
