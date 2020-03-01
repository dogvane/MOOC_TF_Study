using System;
using System.Collections.Generic;
using System.Text;
using MOOC_TF_Study.Common;
using NumSharp;
using Tensorflow;
using static Tensorflow.Binding;

namespace MOOC_TF_Study._6
{
    class _6_3_4
    {
        public static void Run()
        {
            fun1();
        }

        private static void fun1()
        {
            var x_data = np.linspace(-1, 1, 100);
            var y_data = 2 * x_data + 1.0 + np.random.rand(x_data.shape) * 0.4;

            #region 我不要随机数
            y_data = np.array(-0.77710136, -0.54350417, -0.86547821, -0.55992125, -0.837342, -1.33467084,
        -1.2444469, -0.82076245, -1.32381571, -0.34003532, -0.32571077, -0.20681164, -0.5856616, -0.29864504,
        -0.93447063, -0.14173247, -0.32288982, 0.1057742, -0.61476565, -0.25148246, 0.22452511, 0.16641991,
        0.17846703, -0.01658544, -0.01971127, -0.09563799, 0.61697675, -0.26326721, 0.25372583, 0.79789147,
        0.35918527, 0.39203612, -0.41106826, 0.56028576, 1.02617934, -0.10681189, 0.88786259, 1.00214365,
        0.47857213, 0.52466823, 0.48748845, 0.78675198, 1.44487602, 1.07785516, 0.78161655, 0.81507455,
        0.79752379, 1.07105367, 1.21923287, 1.14126001, 1.19698625, 0.77071674, 1.32659001, 1.1879609,
        1.51570375, 1.32938085, 1.04449763, 1.19243454, 2.03804625, 1.22214014, 1.36984456, 1.60431261,
        1.34007491, 1.30594774, 1.31724576, 1.86786032, 0.96547259, 1.65217273, 1.68022047, 1.13190774,
        2.34050031, 2.5374313, 1.98572393, 1.9033125, 1.91291599, 1.65070139, 1.85006016, 2.16014724,
        2.30318022, 1.56625583, 2.26075892, 2.01487454, 2.59661301, 2.50489875, 2.54016046, 2.43119098,
        2.02573928, 2.13201859, 2.33623011, 2.26637834, 3.25502429, 2.92800369, 2.22934641, 2.79055945,
        3.67187637, 2.87554269, 3.46687067, 2.72311618, 3.65135512, 2.95524828);

            #endregion

            var x = tf.placeholder(TF_DataType.TF_DOUBLE, name: "x");
            var y = tf.placeholder(TF_DataType.TF_DOUBLE, name: "y");

            var w = tf.Variable(1.0, name: "w0");
            var b = tf.Variable(0.0, name: "b0");

            var pred = model(x, w, b);

            var train_epochs = 10;
            var learning_rate = (float)0.05;

            var loss_function = tf.reduce_mean(tf.square(y - pred));

            var optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function);

            var display_step = 10;
            var step = 0;
            var loss_list = new List<double>();

            using (var sess = tf.Session())
            {
                var init = tf.global_variables_initializer();
                sess.run(init);

                for (var epoch = 0; epoch < train_epochs; epoch ++ )
                {
                    foreach (var (xs, ys) in zip<double>(x_data, y_data))
                    {
                        var (_, loss) = sess.run((optimizer, loss_function),
                            feed_dict: new[] {new FeedItem(x, xs), new FeedItem(y, ys)});

                        loss_list.Add(loss);
                        step++;
                        if(step % display_step == 0)
                            Console.WriteLine($"Train Epoch: {epoch + 1}, Setp:{ step } loss={((double)loss).ToString("f9")}");
                    }
                }

                var x_test = 3.21;
                var predict = sess.run(pred, feed_dict: new FeedItem(x, x_test));
                Console.WriteLine("预测值:{0}", predict);

                var target = 2 * x_test + 1;
                Console.WriteLine("目标值:{0}", target);
            }
        }

        static Tensor model(Tensor x, Tensor w, Tensor b)
        {
            return tf.multiply(x, w) + b;
        }
    }
}
