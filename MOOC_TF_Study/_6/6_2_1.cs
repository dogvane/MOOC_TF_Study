using System;
using System.Collections.Generic;
using System.Text;
using MOOC_TF_Study.Common;
using NumSharp;
using Tensorflow;
using static Tensorflow.Binding;

namespace MOOC_TF_Study._6
{
    class _6_2_1
    {
        public static void Run()
        {
            fun1();
        }

        private static void fun1()
        {
            np.random.seed(5);
            var x_data = np.linspace(-1, 1, 100);
            var y_data = 2 * x_data + 1.0 + np.random.rand(x_data.shape) * 0.4;

            Console.WriteLine(x_data.ToString());
            Console.WriteLine(y_data.ToString());
        }

    }
}
