using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ch3
{
    class Program
    {
        static int[] step_function(double[] x)
        {
            int[] y = new int[x.Length];
            for (int i = 0; i < y.Length; i++)
            {
                if (x[i] > 0)
                    y[i] = 1;
                else
                    y[i] = 0;
            }
            return y;
        }

        static double[] sigmoid(double[] x)
        {
            double[] y = new double[x.Length];
            for (int i = 0; i < y.Length; i++)
            {
                y[i] = 1.0 / (1.0 + System.Math.Exp(-1 * x[i]));
            }
            return y;
        }


        static void Main(string[] args)
        {
            int[] y = step_function(new double[] { -1.0, 1.0, 2.0 });
            for (int i = 0; i < y.Length; i++)
                System.Console.Write("{0}, ", y[i]);
            System.Console.WriteLine("");

            double[] yy = sigmoid(new double[]{ -1, 1, 2});
            for (int i = 0; i < yy.Length; i++)
                System.Console.Write("{0}, ", yy[i]);
            System.Console.WriteLine("");
        }
    }
}
