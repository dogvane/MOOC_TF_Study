using System;
using System.Collections.Generic;
using System.Text;
using MOOC_TF_Study.Common;
using NumSharp;
using static Tensorflow.Binding;

namespace MOOC_TF_Study._5
{
    class _5_2_1
    {
        public static void Run()
        {
            fun1();
            fun2();
            fun3();
            fun4();
        }

        private static void fun1()
        {
            var tens1 = tf.constant(np.array(1, 2, 3));
            var sess = tf.Session();
            var ret = sess.run(tens1);

            Console.WriteLine(ret.ToString());

            sess.close();
        }

        static void fun2()
        {
            var node1 = tf.constant(3.0, tf.float32, name: "node1");
            var node2 = tf.constant(4.0, tf.float32, name: "node1");

            var result = tf.add(node1, node2);

            using (var sess = tf.Session())
            {
                var ret = sess.run(result);
                Console.WriteLine(ret.ToString());
            }
        }

        static void fun3()
        {
            var node1 = tf.constant(3.0, tf.float32, name: "node1");
            var node2 = tf.constant(4.0, tf.float32, name: "node1");

            var result = tf.add(node1, node2);

            using (var sess = tf.Session())
            {
                sess.as_default();
                
                var ret = result.eval();
                Console.WriteLine(ret.ToString());
            }

            // var sess = tf.InteractiveSession(); c#用不上交互模式，所以这个功能应该没改过来
        }

        static void fun4()
        {
            var node1 = tf.constant(3.0, tf.float32, name: "node1");
            var node2 = tf.constant(4.0, tf.float32, name: "node1");

            var result = tf.add(node1, node2);

            using (var sess = tf.Session())
            {
                try
                {
                    var ret = result.eval();
                    Console.WriteLine(ret.ToString());
                }
                catch (Exception e)
                {
                    Console.WriteLine(e);
                }

                var ret2 = result.eval(session: sess);

                Console.WriteLine(ret2.ToString());
            }
        }

    }
}
