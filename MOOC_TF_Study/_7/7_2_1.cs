using System;
using System.Collections.Generic;
using System.Text;
using MOOC_TF_Study.Common;
using NumSharp;
using Tensorflow;
using static Tensorflow.Binding;

namespace MOOC_TF_Study._7
{
    class _7_2_1
    {
        public static void Run()
        {
            fun1();
        }

        private static void fun1()
        {
            var scalar_value = 18;
            Console.WriteLine($"scalar_value: {scalar_value}");

            var scalar_np = np.array(scalar_value);
            Console.WriteLine($"scalar_np: {scalar_np} {scalar_np.shape}");

            // 向量
            var vector_value = new[] {1, 2, 3};
            var vector_np = np.array(vector_value);

            Console.WriteLine($"vector {vector_np}  {vector_np.Shape}");

            // 矩阵
            var matrix_list = new[,] {{1, 2, 3}, {4, 5, 6}};
            var matrix_np = np.array(matrix_list);

            NDArray matrix_np2 = new[,] { { 1, 2, 3 }, { 4, 5, 6 } };

            Console.WriteLine($"matrix_list:{matrix_list}  matrix_np:{matrix_np} matrix_np.shape={matrix_np.Shape}");

            // 行向量
            var vector_row = np.array(new[,] {{1, 2, 3}});
            Console.WriteLine($"vector_row:{vector_row} shape={vector_row.Shape}");

            // 列向量
            var vector_column = np.array(new[,,] {{{4}, {5}, {6}}});
            Console.WriteLine($"vector_column:{vector_column} shape={vector_column.Shape}");

            Console.WriteLine($"行列转置： vector_row:{vector_row} shape={vector_row.Shape} vector_row.T:{vector_row.T} vector_row.T.shape={vector_row.T.Shape}");

            // 矩阵运算
            var matrix_a = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });

            // 乘法
            var matrix_b = matrix_a * 2;
            Console.WriteLine($"matrix_b: {matrix_b} matrix_b.shape={matrix_b.Shape}");

            // 加法
            var matrix_c = matrix_a + 2;
            Console.WriteLine($"matrix_c: {matrix_c} matrix_c.shape={matrix_c.Shape}");

            // 矩阵间运算

            var matrix_d = np.array(new[,] { { -1, -2, -3 }, { -4, -5, -6 } });

            var matrix_e = matrix_a + matrix_d;
            Console.WriteLine($"matrix_e: {matrix_e} matrix_e.shape={matrix_e.Shape}");
        }

    }
}
