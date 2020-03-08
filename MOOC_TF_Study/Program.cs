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
using System.IO;
using MOOC_TF_Study._10;

namespace MOOC_TF_Study
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            // CheckTFVersion();

            //TestSaverFolder();

            //return;

            _10_4_1.Run();
        }

        static void CheckTFVersion()
        {

            Console.WriteLine(Environment.OSVersion.ToString(), Color.Yellow);
            Console.WriteLine($"TensorFlow Binary v{tf.VERSION}", Color.Yellow);
            Console.WriteLine($"TensorFlow.NET v{Assembly.GetAssembly(typeof(TF_DataType)).GetName().Version}", Color.Yellow);

        }


        static void TestSaverFolder()
        {
            var sess = tf.Session();

            tf.Variable(tf.zeros(10), name: "z");

            var init = tf.global_variables_initializer();
            sess.run(init);

            CheckFolder(sess, "Test"); // ok lastcheckpoint is "Test/2.tf"
            CheckFolder(sess, @"r:\Test"); // ok 
            CheckFolder(sess, @"..\..\Test"); // ok
            CheckFolder(sess, @"r:/Test"); // fail
            CheckFolder(sess, "../../Test"); // fail
        }

        private static void CheckFolder(Session sess, string path)
        {
            if (!Directory.Exists(path))
                Directory.CreateDirectory(path);

            // Console.WriteLine(new FileInfo(path).FullName);

            var saver = tf.train.Saver();

            saver.save(sess, Path.Combine(path, "1.tf"));
            saver.save(sess, Path.Combine(path, "2.tf"));

            var l = tf.train.latest_checkpoint(path);
            Console.WriteLine(l);
        }
    }
}
