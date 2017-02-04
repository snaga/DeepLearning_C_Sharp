using System;
using System.Linq;
using System.Diagnostics;

namespace org.snaga.numeric
{
    public class np
    {
        public static int[] shape(double[,] a)
        {
            return new int[] { a.GetLength(0), a.GetLength(1) };
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
            double[,] c = new double[a.GetLength(0), a.GetLength(1)];

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

        public static double sum(double[] a)
        {
            double b = 0;
            for (int i = 0; i < a.Length; i++)
                b += a[i];
            return b;
        }

        public static double[] sum(double[,] a)
        {
            double[] b = new double[a.GetLength(0)];
            for (int i = 0; i < a.GetLength(0); i++)
                b[i] = sum(row(a, i));
            return b;
        }

        public static double[] exp(double[] a)
        {
            double[] b = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
                b[i] = Math.Exp(a[i]);
            return b;
        }

        public static double dot(double[] a, double[] b)
        {
            return a.Zip(b, (d1, d2) => d1 * d2).Sum();
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

        public static double[] col(double[,] a, int idx)
        {
            double[] aa = new double[a.GetLength(0)];
            for (int i = 0; i < aa.Length; i++)
                aa[i] = a[i, idx];
            return aa;
        }

        public static double[,] dot(double[,] a, double[,] b)
        {
            Debug.Assert(a.GetLength(1) == b.GetLength(0));
            double[,] dot = new double[a.GetLength(0), b.GetLength(1)];

            //            Console.WriteLine("a {0}", a2s(np.shape(a)));
            //            Console.WriteLine("b {0}", a2s(np.shape(b)));

            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < b.GetLength(1); j++)
                {
                    //                    Console.WriteLine("dotproduct {0}", _dot1(_row(a, i), _col(b, j)));
                    dot[i, j] = np.dot(row(a, i), col(b, j));
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
}