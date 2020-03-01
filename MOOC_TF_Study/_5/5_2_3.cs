using System;
using System.Collections.Generic;
using System.Text;
using MOOC_TF_Study.Common;
using NumSharp;
using static Tensorflow.Binding;

namespace MOOC_TF_Study._5
{
    class _5_2_3
    {
        public static void Run()
        {
            fun1();
            fun2();

        }

        private static void fun1()
        {
            tf.reset_default_graph();

            var value = tf.Variable(0, name: "value");
            var one = tf.constant(1);
            var new_value = tf.add(value, one);
            var update_value = tf.assign(value, new_value);

            var init = tf.global_variables_initializer();

            using (var sess = tf.Session())
            {
                sess.run(init);

                for (var i = 0; i < 10; i++)
                {
                    sess.run(update_value);
                    Console.WriteLine(sess.run(value).ToString());
                }

                var logdir = "r:/log";
                var writer = tf.summary.FileWriter(logdir, tf.get_default_graph());
                tf.summary.histogram("value", value);
            }
        }

        private static void fun2()
        {
            tf.reset_default_graph();

            var value = tf.Variable(0, name: "value");

            var init = tf.global_variables_initializer();

            var sum = 0;
            using (var sess = tf.Session())
            {
                sess.run(init);

                for (var i = 1; i <= 10; i++)
                {
                    var newValue = tf.add(value, i); // 通过当前值算出一个新的值

                    var op = tf.assign(value, newValue); // 再将值赋值到原先的值里
                    sess.run(op);   // 迷之操作

                    sum += i;
                    Console.WriteLine($"index{i} .net value:{sum} tf value:{sess.run(value).ToString()}");
                }
            }

            var logdir = "r:/log";
            var writer = tf.summary.FileWriter(logdir, tf.get_default_graph());
            writer.add_summary("");
        }
    }
}
