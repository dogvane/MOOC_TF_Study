using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using MOOC_TF_Study.Common;
using NumSharp;
using Tensorflow;
using Tensorflow.Hub;
using static Tensorflow.Binding;

namespace MOOC_TF_Study._10
{
    class _10_4_1
    {
        public static void Run()
        {
            fun1();
        }

        private static void fun1()
        {
            var testFileName = "../../../../data/cifar-10-batches-bin/data_batch_1.bin";
            var testFilePath = "../../../../data";
            var watch = Stopwatch.StartNew();

            var result = Cifar10ModelLoader.LoadAsync(testFilePath, showProgressInConsole: true).Result;
            Console.WriteLine(watch.Elapsed);

            Console.WriteLine($"result.Train.Data.Shape{result.Train.Data.Shape}");
            Console.WriteLine($"result.Train.Labels.Shape{result.Train.Labels.Shape}");

            // 这里需要引用 Tensorflow.Hub 项目
            var (images, labels) = Cifar10ModelLoader.LoadData(testFileName);
            Console.WriteLine(images.Shape);
            Console.WriteLine(labels.Shape);
        }
    }
}
