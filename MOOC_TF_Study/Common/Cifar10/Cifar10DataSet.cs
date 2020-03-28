using NumSharp;
using NumSharp.Backends.Unmanaged;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Tensorflow.Hub
{
    public class Cifar10DataSet : DataSetBase
    {
        public int NumOfExamples { get; private set; }
        public int EpochsCompleted { get; private set; }
        public int IndexInEpoch { get; private set; }

        public Cifar10DataSet(NDArray images, NDArray labels)
        {
            EpochsCompleted = 0;
            IndexInEpoch = 0;

            NumOfExamples = images.shape[0];

            Data = images;
            Labels = labels;
        }

        /// <summary>
        /// 将数据集的数据做归一化处理
        /// </summary>
        public void NormalizeData()
        {
            // 训练集和数据集的图形数据做归一化
            Data = Data.astype(NPTypeCode.Double) / 255.0;
            Labels = DenseToOneHot(Labels, 10);
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

        public (NDArray, NDArray) GetNextBatch(int batch_size, bool fake_data = false, bool shuffle = true)
        {
            if (IndexInEpoch >= NumOfExamples)
                IndexInEpoch = 0;

            var start = IndexInEpoch;
            // Shuffle for the first epoch
            if (EpochsCompleted == 0 && start == 0 && shuffle)
            {
                var perm0 = np.arange(NumOfExamples);
                np.random.shuffle(perm0);
                Data = Data[perm0];
                Labels = Labels[perm0];
            }

            // Go to the next epoch
            if (start + batch_size > NumOfExamples)
            {
                // Finished epoch
                EpochsCompleted += 1;

                // Get the rest examples in this epoch
                var rest_num_examples = NumOfExamples - start;
                var images_rest_part = Data[np.arange(start, NumOfExamples)];
                var labels_rest_part = Labels[np.arange(start, NumOfExamples)];
                // Shuffle the data
                if (shuffle)
                {
                    var perm = np.arange(NumOfExamples);
                    np.random.shuffle(perm);
                    Data = Data[perm];
                    Labels = Labels[perm];
                }

                start = 0;
                IndexInEpoch = batch_size - rest_num_examples;
                var end = IndexInEpoch;
                var images_new_part = Data[np.arange(start, end)];
                var labels_new_part = Labels[np.arange(start, end)];

                return (np.concatenate(new[] { images_rest_part, images_new_part }, axis: 0),
                    np.concatenate(new[] { labels_rest_part, labels_new_part }, axis: 0));
            }
            else
            {
                IndexInEpoch += batch_size;
                var end = IndexInEpoch;
                return (Data[np.arange(start, end)], Labels[np.arange(start, end)]);
            }
        }
    }
}
