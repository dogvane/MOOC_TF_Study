using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using MOOC_TF_Study.Common;
using NumSharp;
using Tensorflow;
using static Tensorflow.Binding;

namespace MOOC_TF_Study._7
{
    class _7_4_2
    {
        public static void Run()
        {
            fun1();
        }

        static Tensor model(Tensor x, Tensor w, Tensor b)
        {
            return tf.matmul(x, w) + b;
        }

        private static void fun1()
        {
            var csvFileName = "../../../../data/boston.csv";
            Console.WriteLine(new FileInfo(csvFileName).FullName);
            var csvArray = Read_csv_by_double(csvFileName);

            normalization(csvArray);

            var df = np.array(csvArray);
            var x_data = df[":,:12"];
            var y_data = df[":,12"];

            var x = tf.placeholder(TF_DataType.TF_FLOAT, (-1, 12), "X");
            var y = tf.placeholder(TF_DataType.TF_FLOAT, (-1, 1), "Y");

            tf.name_scope("Model");
            var w = tf.Variable(tf.random_normal((12, 1), stddev: 0.01f), name: "W");
            var b = tf.Variable(1.0, dtype:TF_DataType.TF_FLOAT, name: "b");

            var pred = model(x, w, b);

            var learning_rate = 0.01f;
            var train_epochs = 50;

            var loss_function = tf.reduce_mean(tf.pow(y - pred, 2));
            var optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function);

            int[] indexArray = new int[x_data.shape[0]];
            for (var i = 0; i < indexArray.Length; i++)
                indexArray[i] = i;
            
            using (var sess = tf.Session())
            {
                var init = tf.global_variables_initializer();
                sess.run(init);

                for (var i = 0; i < train_epochs; i++)
                {
                    var loss_sum = 0.0;

                    for (var rowIndex = 0; rowIndex < indexArray.Length; rowIndex++)
                    {
                        var index = indexArray[rowIndex];

                        var xs= x_data[$"{index}:{index+1}"];
                        var ys = y_data[$"{index}:{index+1}"];

                        var (_, loss) = sess.run((optimizer, loss_function), new FeedItem(x, xs), new FeedItem(y, ys));

                        loss_sum += loss;
                    }

                    var b0temp = sess.run(b);
                    var w0temp = sess.run(w);

                    var loss_average = loss_sum / indexArray.Length;
                    
                    // indexArray.Shuffle();

                    Console.WriteLine($"epoch={i} loss={loss_average} b={b0temp} w={w0temp}");
                }

                var n = 348;
                var x_test = x_data[$"{n}:{n+1}"];

                var predict = sess.run(pred, new FeedItem(x, x_test));

                Console.WriteLine($"预测值：{predict}");

                var target = y_data[$"{n}:{n + 1}"];
                Console.WriteLine($"标签值：{target}");
            }
        }

        /// <summary>
        /// 对数据做归一化
        /// </summary>
        /// <remarks>
        /// py代码
        /// for i in range(12):
        ///     df[:, i] = df[:, i]/ (df[:, i].max() - df[:, i].min())    
        /// </remarks>
        /// <param name="csvArray"></param>
        private static void normalization(float[][] csvArray)
        {

            for (var y = 0; y < 12; y++)
            {
                var max = 0.0;
                var min = float.MaxValue;

                for (var x = 0; x < csvArray.Length; x++)
                {
                    max = Math.Max(max, csvArray[x][y]);
                    min = Math.Min(min, csvArray[x][y]);
                }

                for (var x = 0; x < csvArray.Length; x++)
                {
                    csvArray[x][y] = (float)(csvArray[x][y] / (max - min));
                }
            }
        }

        static float[][] Read_csv_by_double(string fileName, bool hasHeader = true)
        {
            // 本来是想用CSVHelper，发现它只适合解析出对象，没法方便的解析出数组
            if (!File.Exists(fileName))
            {
                throw new Exception($"not find file. {fileName}");
            }

            var lines = File.ReadAllLines(fileName);
            var start = 0;

            if (hasHeader)
            {
                start = 1;
            }

            List<float[]> datas = new List<float[]>();

            for (var i = start; i < lines.Length; i++)
            {
                if (string.IsNullOrEmpty(lines[i]))
                    continue;

                var array = lines[i].Split(',');
                var newItem = new float[array.Length];
                for(var y = 0; y < array.Length; y++)
                {
                    if (string.IsNullOrEmpty(array[y]))
                        newItem[y] = 0;

                    float temp;
                    if(!float.TryParse(array[y], out temp))
                    {
                        // 展示先不抛出错误了
                        newItem[y] = 0;
                        continue;
                    }

                    newItem[y] = temp;
                }

                datas.Add(newItem);
            }

            return datas.ToArray();
        }
    }
}
