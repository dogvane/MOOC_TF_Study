using System;
using System.Collections.Generic;
using System.Text;
using MOOC_TF_Study.Common;
using NumSharp;
using static Tensorflow.Binding;

namespace MOOC_TF_Study._5
{
    class _5_1_4
    {
        public static void Run()
        {
            tf.reset_default_graph();

            var a = tf.Variable(1, name: "a");
            var b = tf.add(a, 1, name: "b");
            var c = tf.multiply(b, 4, name:"c");
            var d = tf.subtract(c, b, name: "d");

            var logdir = "r:/log";

            var writer = tf.summary.FileWriter(logdir, tf.get_default_graph());
            // 没close
        }
    }
}
