using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using MOOC_TF_Study.Common;
using NumSharp;
using NumSharp.Backends.Unmanaged;
using Tensorflow;
using Tensorflow.Hub;
using static Tensorflow.Binding;

namespace MOOC_TF_Study._10
{
    class _10_4_2
    {
        public static void Run()
        {
            fun1();
        }

        private static void fun1()
        {
            var testFilePath = "../../../../data";
            var result = Cifar10ModelLoader.LoadAsync(testFilePath, showProgressInConsole: true).Result;

            Console.WriteLine($"result.Train.Data.Shape{result.Train.Data.Shape}");
            Console.WriteLine($"result.Train.Labels.Shape{result.Train.Labels.Shape}");

            // 训练集和数据集的图形数据做归一化
            var train_normalize = result.Train.Data.astype(NPTypeCode.Double) / 255.0;
            var testnormalize = result.Test.Data.astype(NPTypeCode.Double) / 255.0;

            var train_onehot = DenseToOneHot(result.Train.Labels, 10);
            var test_onehot = DenseToOneHot(result.Test.Labels, 10);

            Console.WriteLine("train_normalize：{0}", train_normalize[0].ToString());
            Console.WriteLine("train_lables:{0}", result.Train.Labels["0:5"].ToString());
            Console.WriteLine("train_onehot:{0}", train_onehot["0:5"].ToString());
        }

        private static NDArray DenseToOneHot(NDArray labels_dense, int num_classes)
        {
            int stop = labels_dense.shape[0];
            NDArray ndArray1 = np.arange(stop) * num_classes;
            NDArray ndArray2 = np.zeros(stop, num_classes);
            ArraySlice<byte> arraySlice = labels_dense.Data<byte>();
            for (int index = 0; index < stop; ++index)
            {
                byte num = arraySlice[index];
                ndArray2.SetData((NDArray)1.0, index, (int)num);
            }
            return ndArray2;
        }
    }
}
