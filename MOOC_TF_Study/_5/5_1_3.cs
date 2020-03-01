using System;
using MOOC_TF_Study.Common;
using NumSharp;
using static Tensorflow.Binding;

namespace MOOC_TF_Study._5
{
    internal class _5_1_3
    {
        public static void Run()
        {
            fun1();
            fun2();
            fun3();
            fun4();
            fun5();
        }

        private static void fun1()
        {
            Console.WriteLine();
            Console.WriteLine("fun1");
            Console.WriteLine();

            var arr = np.array(new[]
            {
                new[] {new[] {1, 2, 2}, new[] {2, 2, 3}},
                new[] {new[] {3, 5, 6}, new[] {5, 4, 3}},
                new[] {new[] {7, 0, 1}, new[] {9, 1, 9}},
                new[] {new[] {11, 12, 7}, new[] {1, 3, 14}}
            });

            var tens1 = tf.constant(arr, name: "tens1");

            Console.WriteLine(tens1);
        }

        static void fun2()
        {
            Console.WriteLine();
            Console.WriteLine("fun2");
            Console.WriteLine();

            var scalar = tf.constant(100);
            var vector = tf.constant(np.array(new[] {1, 2, 3, 4, 5})); // var vector = tf.constant(np.array(1, 2, 3, 4, 5));
            var matrix = tf.constant(np.array(new[] {new[] {1, 2, 3}, new[] {4, 5, 6}}));
            var cube_matrix = tf.constant(np.array(new[]
            {
                new[] {new[] {1}, new[] {2}, new[] {3}},
                new[] {new[] {4}, new[] {5}, new[] {6}},
                new[] {new[] {7}, new[] {8}, new[] {9}},
            }));

            Console.WriteLine(scalar.shape.Dump());
            Console.WriteLine(vector.shape.Dump());
            Console.WriteLine(matrix.shape.Dump());

            Console.WriteLine(cube_matrix.shape.Dump());
        }

        static void fun3()
        {
            Console.WriteLine();
            Console.WriteLine("fun3");
            Console.WriteLine();


            var array = new[]
            {
                new[] {new[] {1, 2}, new[] {2, 3}},
                new[] {new[] {3, 4}, new[] {5, 6}}
            };

            var tens1 = tf.constant(np.array(array));
            var sess = tf.Session();
            var ret = sess.run(tens1);

            Console.WriteLine(ret.ToString());
            Console.WriteLine(ret.GetData(1, 1, 0).ToString());

            sess.close();
        }

        static void fun4()
        {
            Console.WriteLine();
            Console.WriteLine("fun4");
            Console.WriteLine();

            try
            {
                var a = tf.constant(np.array(new[] { 1, 2 }));
                var b = tf.constant(np.array(2.0, 3.0));

                var result = a + b;
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
            }
        }


        static void fun5()
        {
            Console.WriteLine();
            Console.WriteLine("fun5");
            Console.WriteLine();

            var a = tf.constant(np.array(1, 2), name:"a");
            var b = tf.constant(np.array(2, 3), name:"b");

            var result = a + b;
            Console.WriteLine($"result: {result}");

            var result2 = tf.add(a, b);
            Console.WriteLine($"result2: {result2}");
        }
    }
}
