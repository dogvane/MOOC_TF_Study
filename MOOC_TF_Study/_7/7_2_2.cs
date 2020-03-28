using System;
using System.Collections.Generic;
using System.Text;
using MOOC_TF_Study.Common;
using NumSharp;
using Tensorflow;
using static Tensorflow.Binding;

namespace MOOC_TF_Study._7
{
    class _7_2_2
    {
        public static void Run()
        {
            fun1();
            fun2();
            fun3();
        }

        private static void fun1()
        {
            // 矩阵运算
            var matrix_a = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var matrix_b = np.array(new[,] { { -1, -2, -3 }, { -4, -5, -6 } });

            // 加法
            var matrix_c = matrix_a + matrix_b;
            Console.WriteLine($"matrix_c: {matrix_c} matrix_c.shape={matrix_c.Shape}");
        }

        private static void fun2()
        {
            // 矩阵运算
            var matrix_a = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var matrix_b = np.array(new[,] { { -1, -2, -3 }, { -4, -5, -6 } });

            // 乘法
            var matrix_c = matrix_a * matrix_b;
            Console.WriteLine($"matrix_c: {matrix_c} matrix_c.shape={matrix_c.Shape}");

            var matrix_d = np.multiply(matrix_a, matrix_b);
            Console.WriteLine($"matrix_d: {matrix_d} matrix_d.shape={matrix_d.Shape}");
        }

        private static void fun3()
        {
            // 矩阵运算
            var matrix_a = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var matrix_b = np.array(new[,] { { 1, 2, 3, 4 }, { 2, 1, 2, 0 }, { 3, 4, 1, 2 } });

            var matrix_d = np.matmul(matrix_a, matrix_b);
            Console.WriteLine($"matrix_d: {matrix_d} matrix_d.shape={matrix_d.Shape}");
        }
    }
}
