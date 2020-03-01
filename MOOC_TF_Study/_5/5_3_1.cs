using System;
using System.Collections.Generic;
using System.Text;
using MOOC_TF_Study.Common;
using NumSharp;
using Tensorflow;
using static Tensorflow.Binding;

namespace MOOC_TF_Study._5
{
    class _5_3_1
    {
        public static void Run()
        {
            fun1();
        }

        private static void fun1()
        {
            tf.reset_default_graph();

            using (var sess = tf.Session())
            {
                // var input1 = tf.constant(np.array(1.0, 2.0, 3.0), dtype: tf.float32, name: "input1");
                var input1 = tf.constant(np.array((float)1.0, (float)2.0, (float)3.0), dtype: tf.float16, name: "input1");
                var input2 = tf.Variable(tf.random_uniform(3), name: "input2");

                // var output = tf.add_n(new[] {input1, input2 }, name: "add");
                // 目前版本的数组模式，只支持2个张量，而input1和input2 ，一个是张量，一个是变量
                var output = tf.add(input1, input2, name: "add");

                var init = tf.global_variables_initializer();
                sess.run(init);
                
                // var writer = tf.summary.FileWriter(logdir, tf.get_default_graph());
                var logdir = @"r:/log";
                var writer = tf.summary.FileWriter(logdir, sess.graph);
                tf.summary.scalar("input", input1);
            }
        }
    }
}
