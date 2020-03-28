using System;
using System.Collections.Generic;
using System.Text;
using MOOC_TF_Study.Common;
using NumSharp;
using Tensorflow;
using static Tensorflow.Binding;

namespace MOOC_TF_Study._7
{
    class _7_2_3
    {
        public static void Run()
        {
            fun1();
        }

        private static void fun1()
        {
            var matrix_a = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var matrix_b = (NDArray) new[,] {{1}, {2}, {3}};

            var matrix_d = np.matmul(matrix_a, matrix_b);

            Console.WriteLine($"matrix_d: {matrix_d} matrix_d.shape={matrix_d.Shape}");
            Console.WriteLine();

            matrix_a = new[,] {{1, 2, 3}};
            matrix_b = new[,] {{2}, {4}, {-1}};

            matrix_d = np.matmul(matrix_a, matrix_b);

            Console.WriteLine($"matrix_d: {matrix_d} matrix_d.shape={matrix_d.Shape}");
            Console.WriteLine();

            // 行列转置
            var vector_row = (NDArray) new[,] {{1, 2, 3}};
            Console.WriteLine($"vector_row:{vector_row} shape={vector_row.Shape}");
            Console.WriteLine($"{vector_row.T} .Tshape={vector_row.T.Shape}");

            var vector_column = vector_row.reshape(3, 1);
            Console.WriteLine($"vector_column {vector_column} vector_column.T {vector_column.T}");
        }

        private static void fun2()
        {
            var matrix_a = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            Console.WriteLine($"vector_row:{matrix_a} shape={matrix_a.Shape}");
            Console.WriteLine($"{matrix_a.T} .Tshape={matrix_a.T.Shape}");
        }
    }
}
