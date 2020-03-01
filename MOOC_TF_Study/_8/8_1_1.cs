using System;
using System.Collections.Generic;
using System.Text;
using MOOC_TF_Study.Common;
using NumSharp;
using Tensorflow;
using Tensorflow.Hub;
using static Tensorflow.Binding;

namespace MOOC_TF_Study._8
{
    class _8_1_1
    {
        public static void Run()
        {
            fun1();
        }

        private static void fun1()
        {
            var path = "../../../../data/MNIST_data/";
            // 这里需要引用 Tensorflow.Hub 项目
            var mnist = MnistModelLoader.LoadAsync(path, oneHot:true).Result;

            Console.WriteLine($"训练集 train 数量:{mnist.Train.NumOfExamples}");
            Console.WriteLine($"验证集 validation 数量:{mnist.Validation.NumOfExamples}");
            Console.WriteLine($"测试集 test 数量:{mnist.Test.NumOfExamples}");

            Console.WriteLine($"train images shape: {mnist.Train.Data.Shape}");
            Console.WriteLine($"labels shape:{ mnist.Train.Labels.Shape}");

            var image = mnist.Train.Data[0];

            Console.WriteLine($"image shape={image.Shape}");
            Console.WriteLine($"image data={image.ToString()}");
            Console.WriteLine($"image.reshape={image.reshape(28,28)}");

            // 画图是不可能画图了的。

        }

    }
}
