using System;
using System.Collections.Generic;
using System.Text;
using MOOC_TF_Study.Common;
using NumSharp;
using static Tensorflow.Binding;

namespace MOOC_TF_Study._5
{
    class _5_2_2
    {
        public static void Run()
        {
            fun1();
            fun2();
        }

        private static void fun1()
        {
            var node1 = tf.Variable(3.0, dtype: tf.float32, name: "node1");
            var node2 = tf.Variable(4.0, dtype: tf.float32, name: "node2");

            var result = tf.add(node1, node2, name: "add");

            using (var sess = tf.Session())
            {
                // 变量初始化
                var init = tf.global_variables_initializer();
                sess.run(init);

                var ret = sess.run(result);
                Console.WriteLine(ret.ToString());
            }
        }

        private static void fun2()
        {
            var node1 = tf.Variable(3.0, dtype: tf.float32, name: "node1");
            var node2 = tf.Variable(4.0, dtype: tf.float32, name: "node2");

            var result = tf.add(node1, node2, name: "add");

            using (var sess = tf.Session())
            {
                // 变量初始化
                var init = tf.global_variables_initializer();

                try
                {
                    var ret = sess.run(result);
                    Console.WriteLine(ret.ToString());
                }
                catch (Exception e)
                {
                    Console.WriteLine(e);
                }
            }
        }
    }
}
