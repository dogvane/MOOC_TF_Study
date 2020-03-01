using System;
using System.Collections.Generic;
using System.Text;
using MOOC_TF_Study.Common;
using NumSharp;
using Tensorflow;
using static Tensorflow.Binding;

namespace MOOC_TF_Study._6
{
    class _6_2_3
    {
        public static void Run()
        {
            fun1();
        }

        private static void fun1()
        {
            var x = tf.placeholder(TF_DataType.TF_FLOAT, name: "x");
            var y = tf.placeholder(TF_DataType.TF_FLOAT, name: "y");

            var w = tf.Variable((float)1.0, name: "w0");
            var b = tf.Variable((float)0.0, name: "b0");

            var pred = model(x, w, b);

        }

        static Tensor model(Tensor x, Tensor w, Tensor b)
        {
            return tf.multiply(x, w) + b;
        }
    }
}
