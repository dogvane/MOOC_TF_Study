using System;
using System.Collections.Generic;
using System.Text;
using MOOC_TF_Study.Common;
using NumSharp;
using Tensorflow;
using static Tensorflow.Binding;

namespace MOOC_TF_Study._5
{
    class _5_2_4
    {
        public static void Run()
        {
            fun1();
            fun2();
        }

        private static void fun1()
        {
            var a = tf.placeholder(tf.float32, name: "a");
            var b = tf.placeholder(tf.float32, name: "b");
            var c = tf.multiply(a, b, name: "c");

            var init = tf.global_variables_initializer();

            using (var sess = tf.Session())
            {
                sess.run(init);
                var result = sess.run(c, feed_dict: new[] {new FeedItem(a, 8.0), new FeedItem(b, 3.5)});
                Console.WriteLine(result.ToString());
            }
        }

        private static void fun2()
        {
            var a = tf.placeholder(tf.float32, name: "a");
            var b = tf.placeholder(tf.float32, name: "b");
            var c = tf.multiply(a, b, name: "c");
            var d = tf.subtract(a, b, name: "d");

            var init = tf.global_variables_initializer();

            using (var sess = tf.Session())
            {
                sess.run(init);

                var result = sess.run(new[] {c, d},
                    feed_dict: new[]
                    {
                        new FeedItem(a, np.array(8.0, 2.0, 3.5)),
                        new FeedItem(b, np.array(1.5, 2.0, 4.0))
                    });
                Console.WriteLine(result.Dump());
            }
        }
    }
}
