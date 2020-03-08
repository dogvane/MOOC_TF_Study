using System;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.IO;
using NumSharp;
using ICSharpCode.SharpZipLib.GZip;
using ICSharpCode.SharpZipLib.Tar;
using System.Threading;

namespace Tensorflow.Hub
{
    public class Cifar10ModelLoader : IModelLoader<Cifar10DataSet>
    {
        private const string DEFAULT_SOURCE_URL = "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
        private const string DEFAULT_LOCAL_DOWNLOAD_FILENAME = "cifar-10-binary.tar.gz";

        private readonly string[] TRAIN_FILE = { "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin" };
        private const string TEST_FILE = "test_batch.bin";

        /// <summary>
        /// 
        /// </summary>
        /// <param name="trainDir"></param>
        /// <returns></returns>
        public bool CheckFileExists(string trainDir)
        {
            foreach(var name in TRAIN_FILE)
            {
                var fileName = Path.Combine(trainDir, name);
                if (!File.Exists(fileName))
                    return false;
            }

            var testFileName = Path.Combine(trainDir, TEST_FILE);
            return File.Exists(testFileName);
        }

        async Task DownFiles(string url, string trainDir, bool showProcess)
        {
            var localFileName = Path.Combine(trainDir, DEFAULT_LOCAL_DOWNLOAD_FILENAME);
            if(!File.Exists(localFileName))
            {
                Console.WriteLine("begin down file.");
                // local file exits, unzip local file.
                await this.DownloadAsync(url, trainDir, DEFAULT_LOCAL_DOWNLOAD_FILENAME, showProcess);
            }

            ExtractTGZ(localFileName, trainDir);
        }

        public static void ExtractTGZ(string gzArchiveName, string destFolder)
        {
            var flag = gzArchiveName.Split(Path.DirectorySeparatorChar).Last().Split('.').First() + ".bin";
            if (File.Exists(Path.Combine(destFolder, flag))) return;

            Console.WriteLine($"Extracting.");
            var task = Task.Run(() =>
            {
                using (var inStream = File.OpenRead(gzArchiveName))
                {
                    using (var gzipStream = new GZipInputStream(inStream))
                    {
                        using (TarArchive tarArchive = TarArchive.CreateInputTarArchive(gzipStream))
                            tarArchive.ExtractContents(destFolder);
                    }
                }
            });

            while (!task.IsCompleted)
            {
                Thread.Sleep(200);
                Console.Write(".");
            }

            File.Create(Path.Combine(destFolder, flag));
            Console.WriteLine("");
            Console.WriteLine("Extracting is completed.");
        }

        public static async Task<Datasets<Cifar10DataSet>> LoadAsync(string trainDir, bool oneHot = false, int? trainSize = null, int? validationSize = null, int? testSize = null, bool showProgressInConsole = false)
        {
            var setting = new ModelLoadSetting
            {
                TrainDir = trainDir,
                OneHot = oneHot,
                ShowProgressInConsole = showProgressInConsole
            };

            if (trainSize.HasValue)
                setting.TrainSize = trainSize.Value;

            if (validationSize.HasValue)
                setting.ValidationSize = validationSize.Value;

            if (testSize.HasValue)
                setting.TestSize = testSize.Value;

            var loader = new Cifar10ModelLoader();

            return await loader.LoadAsync(setting);
        }

        public async Task<Datasets<Cifar10DataSet>> LoadAsync(ModelLoadSetting setting)
        {
            if (!CheckFileExists(setting.TrainDir))
            {
                if (String.IsNullOrEmpty(setting.SourceUrl))
                    setting.SourceUrl = DEFAULT_SOURCE_URL;

                await DownFiles(setting.SourceUrl, setting.TrainDir, setting.ShowProgressInConsole);
            }

            var trainFiles = TRAIN_FILE.Select(o => Path.Combine(setting.TrainDir, "cifar-10-batches-bin", o)).ToArray();
            var (train_images, train_labels) = LoadData(trainFiles);

            var testBinFile = Path.Combine(setting.TrainDir, "cifar-10-batches-bin", TEST_FILE);
            var (test_images, test_labels) = LoadData(testBinFile);

            var train = new Cifar10DataSet(train_images, train_labels);
            var test = new Cifar10DataSet(test_images, test_labels);

            return new Datasets<Cifar10DataSet>(train, null, test);
        }

        public static (List<byte[]> images, List<byte> labels) LoadByteData(string file)
        {
            Console.WriteLine($"loading data file:{file}");
            // file format: <1 x label><3072 x pixel>
            int bufferlen = 3072;

            List<byte> labels = new List<byte>();
            List<byte[]> images = new List<byte[]>();

            using (var stream = new FileStream(file, FileMode.Open))
            {
                var labelBuff = new byte[1];
                while (true)
                {
                    var readLen = stream.Read(labelBuff, 0, 1);
                    if (readLen == 0)
                        break;

                    var buffer = new byte[bufferlen];
                    readLen = stream.Read(buffer, 0, bufferlen);
                    if (readLen == 0)
                        break;

                    images.Add(buffer);
                    labels.Add(labelBuff[0]);
                }
            }

            return (images, labels);
        }

        public static (NDArray images, NDArray labels) LoadData(params string[] files)
        {
            List<byte> labels = new List<byte>();
            List<byte[]> images = new List<byte[]>();

            foreach (var file in files)
            {
                var (i, l) = LoadByteData(file);
                labels.AddRange(l);
                images.AddRange(i);
            }

            var retImages = np.array<byte>(images.ToArray());

            // 调整数据的结构为 BCWH，相当于把一个，把一个一位数组，转置为3维数组
            retImages = retImages.reshape(images.Count, 3, 32, 32);

            // tf处理图像数据的结构是 BWHC
            // 把通道数据C移动到最后一个维度
            retImages = retImages.transpose(new[] { 0, 2, 3, 1 });

            var retLabels = np.array<byte>(labels);

            return (retImages, retLabels);
        }

        private NDArray DenseToOneHot(NDArray labels_dense, int num_classes)
        {
            var num_labels = labels_dense.shape[0];
            var index_offset = np.arange(num_labels) * num_classes;
            var labels_one_hot = np.zeros(num_labels, num_classes);
            var labels = labels_dense.Data<byte>();
            for (int row = 0; row < num_labels; row++)
            {
                var col = labels[row];
                labels_one_hot.SetData(1.0, row, col);
            }

            return labels_one_hot;
        }
    }
}
