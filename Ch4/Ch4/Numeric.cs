using System;
using System.Linq;
using System.Diagnostics;

using Troschuetz.Random;

namespace org.snaga.numeric
{
    // -----------------------------------------
    // numpy porting functions
    // -----------------------------------------
    public class np
    {
        public static double[] zeros(int size)
        {
            double[] a = new double[size];
            for (int i = 0; i < a.Length; i++)
                a[i] = 0;
            return a;
        }

        public static double[,] zeros_like(double[,] a)
        {
            double[,] b = new double[a.GetLength(0), a.GetLength(1)];
            for (int i = 0; i < a.GetLength(0); i++)
                for (int j = 0; j < a.GetLength(1); j++)
                    b[i, j] = 0;
            return b;
        }

        public static double[] arange(double min, double max, double delta)
        {
            int len = (int)System.Math.Ceiling((max - min) / delta) + 1;
            double[] a = new double[len];
            for (int i = 0; (min + delta * i) <= max; i++)
            {
                a[i] = min + delta * i;
            }
            return a;
        }

        public static int[] shape(double[,] a)
        {
            return new int[] { a.GetLength(0), a.GetLength(1) };
        }

        public static int[] shape(double[] a)
        {
            return new int[] { 0, a.GetLength(0) };
        }

        public static double[] add(double[] a, double b)
        {
            double[] c = new double[a.Length];

            for (int i = 0; i < a.Length; i++)
                c[i] = a[i] + b;
            return c;
        }

        public static double[,] add(double[,] a, double b)
        {
            double[,] c = np.zeros_like(a);

            for (int i = 0; i < a.GetLength(0); i++)
                np.row(c, i, add(row(a, i), b));
            return c;
        }

        public static double[] add(double[] a, double[] b)
        {
            Debug.Assert(a.GetLength(0) == b.GetLength(0));
            double[] c = new double[a.GetLength(0)];

            for (int i = 0; i < a.GetLength(0); i++)
                c[i] = a[i] + b[i];
            return c;
        }

        public static double[,] add(double[,] a, double[,] b)
        {
            Debug.Assert(a.GetLength(0) == b.GetLength(0) || b.GetLength(0) == 1);
            Debug.Assert(a.GetLength(1) == b.GetLength(1));
            double[,] c = new double[a.GetLength(0), a.GetLength(1)];

            for (int i = 0; i < a.GetLength(0); i++)
                if (b.GetLength(0) == 1)
                    np.row(c, i, add(row(a, i), row(b, 0)));
                else
                    np.row(c, i, add(row(a, i), row(b, i)));
            return c;
        }

        public static int argmax(double[] x)
        {
            int max_idx = 0;
            double max_val = x[0];
            for (int i = 1; i < x.Length; i++)
            {
                if (x[i] > max_val)
                {
                    max_idx = i;
                    max_val = x[i];
                }
            }
            return max_idx;
        }

        public static double[] col(double[,] a, int idx)
        {
            double[] aa = new double[a.GetLength(0)];
            for (int i = 0; i < aa.Length; i++)
                aa[i] = a[i, idx];
            return aa;
        }

        public static double[] subtract(double[] a, double[] b)
        {
            Debug.Assert(a.GetLength(0) == b.GetLength(0));
            double[] c = new double[a.GetLength(0)];

            for (int i = 0; i < a.GetLength(0); i++)
                c[i] = a[i] - b[i];
            return c;
        }

        public static double[] pow(double[] a, double b)
        {
            double[] c = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
                c[i] = Math.Pow(a[i], b);
            return c;
        }

        public static double[] log(double[] a)
        {
            double[] c = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
                c[i] = Math.Log(a[i]);
            return c;
        }

        public static double[,] log(double[,] a)
        {
            double[,] c = new double[a.GetLength(0), a.GetLength(1)];
            for (int i = 0; i < a.GetLength(0); i++)
                np.row(c, i, log(row(a, i)));
            return c;
        }

        public static double[] multi(double[] a, double b)
        {
            double[] c = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
                c[i] = a[i] * b;
            return c;
        }

        public static double[] multi(double[] a, double[] b)
        {
            double[] c = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
                c[i] = a[i] * b[i];
            return c;
        }

        public static double[,] multi(double[,] a, double[,] b)
        {
            double[,] c = new double[a.GetLength(0), a.GetLength(1)];
            for (int i = 0; i < a.GetLength(0); i++)
                row(c, i, multi(row(a, i), row(b, i)));
            return c;
        }

        public static double[,] multi(double[,] a, double b)
        {
            double[,] c = np.zeros_like(a);
            for (int i = 0; i < a.GetLength(0); i++)
                for (int j = 0; j < a.GetLength(1); j++)
                    c[i, j] = a[i, j] * b;
            return c;
        }

        public static double[] div(double[] a, double b)
        {
            double[] c = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
                c[i] = a[i] / b;
            return c;
        }

        public static double max(double[] a)
        {
            double max = a[0];
            for (int i = 1; i < a.Length; i++)
                max = Math.Max(max, a[i]);
            return max;
        }

        public static double sum(double[] a)
        {
            double b = 0;
            for (int i = 0; i < a.Length; i++)
                b += a[i];
            return b;
        }

        public static double sum(double[,] a)
        {
            double b = 0;
            for (int i = 0; i < a.GetLength(0); i++)
                b += sum(np.row(a, i));
            return b;
        }

        public static double[] exp(double[] a)
        {
            double[] b = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
                b[i] = Math.Exp(a[i]);
            return b;
        }

        public static double[] row(double[,] a, int idx)
        {
            double[] aa = new double[a.GetLength(1)];
            for (int i = 0; i < aa.Length; i++)
                aa[i] = a[idx, i];
            return aa;
        }

        public static double[,] row(double[,] x, int i, double[] r)
        {
            for (int j = 0; j < r.Length; j++)
                x[i, j] = r[j];
            return x;
        }

        public static double[,] rows(double[,] a, int offset, int len)
        {
            double[,] aa = new double[len, a.GetLength(1)];
            for (int i = 0; i < len; i++)
                for (int j = 0; j < a.GetLength(1); j++)
                    aa[i, j] = a[offset + i, j];
            return aa;
        }

        public static double[,] rows(double[,] a, int[] mask)
        {
            double[,] aa = new double[mask.Length, a.GetLength(1)];
            for (int i = 0; i < mask.Length; i++)
                for (int j = 0; j < a.GetLength(1); j++)
                    aa[i, j] = a[mask[i], j];
            return aa;
        }

        public static double[,] dot(double[,] a, double[,] b)
        {
            int a0 = a.GetLength(0);
            int a1 = a.GetLength(1);
            int b0 = b.GetLength(0);
            int b1 = b.GetLength(1);
            Console.WriteLine("b0 = {0}", b0);
            Debug.Assert(a1 == b0);

            double[,] dot = new double[a0, b1];

            for (int i = 0; i < a0; i++)
            {
                for (int j = 0; j < b1; j++)
                {
                    for (int k = 0; k < b0; k++)
                    {
                        dot[i, j] += a[i, k] * b[k, j];
                    }
                }
            }
            return dot;
        }

        public static double[,] T(double[,] x)
        {
            double[,] t = new double[x.GetLength(1), x.GetLength(0)];
            for (int i = 0; i < x.GetLength(0); i++)
                for (int j = 0; j < x.GetLength(1); j++)
                    t[j, i] = x[i, j];
            return t;
        }

        public static double[,] reshape(double[] x, int rows, int cols)
        {
            double[,] y = new double[rows, cols];
            for (int i = 0; i < x.Length; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    for (int k = 0; k < cols; k++)
                    {
                        y[j, k] = x[i];
                        i++;
                    }
                }
            }

            return y;
        }

        public static int[] random_choice(int a, int b)
        {
            Random rand = new Random();
            int[] aa = new int[b];
            for (int i = 0; i < b; i++)
                aa[i] = rand.Next(a);
            return aa;
        }

        public static double[,] random_randn(int a, int b)
        {
            double[] x;
            var trand = new TRandom();
            x = trand.NormalSamples(0.0, 1.0).Take(a * b).ToArray<double>();
            return reshape(x, a, b);
        }

        public static double[,] random_rand(int a, int b)
        {
            double[] x;
            var trand = new TRandom();
            x = trand.ContinuousUniformSamples(0, 1).Take(a * b).ToArray<double>();
            return reshape(x, a, b);
        }

        public static String str(double[] x)
        {
            String s = "[ ";
            for (int i = 0; i < x.Length; i++)
            {
                if (i > 0)
                    s = s + ", ";
                //                s = s + Math.Round(x[i], 2).ToString();
                s = s + x[i].ToString();
            }
            s = s + " ]";
            return s;
        }
        public static String str(int[] x)
        {
            double[] a = new double[x.Length];
            for (int i = 0; i < a.Length; i++)
                a[i] = (double)x[i];
            return str(a);
        }

        public static String str(double[,] x)
        {
            String s = "[ ";
            for (int i = 0; i < x.GetLength(0); i++)
            {
                double[] xx = new double[x.GetLength(1)];
                for (int j = 0; j < x.GetLength(1); j++)
                    xx[j] = x[i, j];
                if (i > 0)
                    s = s + ",\n  ";
                s = s + str(xx);
            }
            s = s + " ]";
            return s;
        }
    }

    
    // -----------------------------------------
    // Utility functions for neural network
    // -----------------------------------------
    public class nn
    {
        public static double[] sigmoid(double[] x)
        {
            double[] y = new double[x.Length];
            for (int i = 0; i < y.Length; i++)
            {
                y[i] = 1.0 / (1.0 + System.Math.Exp(-1 * x[i]));
            }
            return y;
        }

        // sigmoid() のバッチ対応版
        public static double[,] sigmoid(double[,] x)
        {
            double[,] ss = np.zeros_like(x);
            for (int i = 0; i < x.GetLength(0); i++)
                for (int j = 0; j < x.GetLength(1); j++)
                    ss[i, j] = 1.0 / (1.0 + System.Math.Exp(-1 * x[i,j]));
            return ss;
        }

        public static double[] softmax(double[] a)
        {
            double c = np.max(a);
            double[] exp_a = np.exp(np.add(a, -1.0 * c));
            double[] y = np.div(exp_a, np.sum(exp_a));
            return y;
        }

        // softmax() のバッチ対応版
        public static double[,] softmax(double[,] a)
        {
            double[,] y = np.zeros_like(a);
            for (int i = 0; i < a.GetLength(0); i++)
                np.row(y, i, softmax(np.row(a, i)));
            return y;
        }

        // 2乗和誤差
        public static double mean_squared_error(double[] y, double[] t)
        {
            return 0.5 * np.sum(np.pow(np.subtract(y, t), 2));
        }

        // 交差エントロピー誤差
        public static double cross_entropy_error(double[] y, double[] t)
        {
            double delta = 1e-7;
            return -1.0 * np.sum(np.multi(t, np.log(np.add(y, delta))));
        }

        // cross_entropy_error() のバッチ対応版
        public static double cross_entropy_error(double[,] y, double[,] t)
        {
            Debug.Assert(y.GetLength(0) == t.GetLength(0));
            Debug.Assert(y.GetLength(1) == t.GetLength(1));
            double delta = 1e-7;

            int batch_size = y.GetLength(0);

            double[,] yy = np.add(y, delta);

//            Console.WriteLine("np.log(y) = {0}", np.str(np.log(y)));
//            Console.WriteLine("np.log(yy) = {0}", np.str(np.log(yy)));

            double[,] tmp = np.multi(t, np.log(yy));
//            Console.WriteLine("tmp = {0}", np.str(tmp));

            return np.sum(tmp) / batch_size * -1;
        }

    }
}