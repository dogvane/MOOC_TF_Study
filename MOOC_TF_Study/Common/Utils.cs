using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace MOOC_TF_Study.Common
{
    public static class Utils
    {
        /// <summary>
        /// 把数组内的元素显示出来
        /// </summary>
        /// <param name="arr"></param>
        /// <returns></returns>
        public static string Dump(this int[] arr)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("(");
            foreach (var item in arr)
            {
                sb.Append(item);
                sb.Append(",");
            }

            if (sb.Length > 2)
            {
                sb[sb.Length - 1] = ')';
            }
            else
            {
                sb.Append(")");
            }

            return sb.ToString();
        }

        public static string Dump(this NDArray[] arr)
        {
            StringBuilder sb = new StringBuilder();
            
            foreach (var item in arr)
            {
                sb.AppendLine(item.ToString());
            }

            return sb.ToString();
        }

        static Random rand = new Random();

        /// <summary>
        /// 数组洗牌
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="array"></param>
        /// <returns></returns>
        public static T[] Shuffle<T>(this T[] array)
        {

            int n = array.Length;
            while (n > 1)
            {
                int k = rand.Next(n--);
                T temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
            return array;
        }
    }
}
