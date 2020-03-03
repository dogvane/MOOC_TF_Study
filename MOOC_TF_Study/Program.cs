using System;
using System.Drawing;
using System.Linq;
using System.Reflection;
using MOOC_TF_Study._5;
using MOOC_TF_Study._6;
using MOOC_TF_Study._7;
using MOOC_TF_Study._8;
using Tensorflow;
using Console = Colorful.Console;
using static Tensorflow.Binding;
using MOOC_TF_Study._9;

namespace MOOC_TF_Study
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            CheckTFVersion();

            _9_1_4.Run();
        }

        static void CheckTFVersion()
        {

            Console.WriteLine(Environment.OSVersion.ToString(), Color.Yellow);
            Console.WriteLine($"TensorFlow Binary v{tf.VERSION}", Color.Yellow);
            Console.WriteLine($"TensorFlow.NET v{Assembly.GetAssembly(typeof(TF_DataType)).GetName().Version}", Color.Yellow);

        }
    }
}
