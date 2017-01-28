using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

namespace Ch3
{
    class Program
    {
        static double[] _arange(double min, double max, double delta)
        {
            int len = (int)System.Math.Ceiling((max - min) / delta) + 1;
            double[] a = new double[len];
            for (int i = 0; (min+delta*i) <= max; i++)
            {
                a[i] = min + delta * i;
            }
            return a;
        }

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

        static double[] relu(double[] x)
        {
            double[] a = new double[x.Length];
            for (int i = 0; i < a.Length; i++)
            {
                a[i] = System.Math.Max(0, x[i]);
            }
            return a;
        }

        static String a2s(double[] x)
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
        static String a2s(int[] x)
        {
            double[] a = new double[x.Length];
            for (int i = 0; i < a.Length; i++)
                a[i] = (double)x[i];
            return a2s(a);
        }

        static String a2s(double[,] x)
        {
            String s = "[ ";
            for (int i = 0; i < x.GetLength(0); i++)
            {
                double[] xx = new double[x.GetLength(1)];
                for (int j = 0; j < x.GetLength(1); j++)
                    xx[j] = x[i, j];
                if (i > 0)
                    s = s + ",\n  ";
                s = s + a2s(xx);
            }
            s = s + " ]";
            return s;
        }

        static int[] _shape(double[,] a)
        {
            return new int[] { a.GetLength(0), a.GetLength(1) };
        }

        static double _dot1(double[] a, double[] b)
        {
            return a.Zip(b, (d1, d2) => d1 * d2).Sum();
        }

        static double[] _row(double[,] a, int idx)
        {
            double[] aa = new double[a.GetLength(1)];
            for (int i = 0; i < aa.Length; i++)
                aa[i] = a[idx, i];
            return aa;
        }

        static double[] _col(double[,] a, int idx)
        {
            double[] aa = new double[a.GetLength(0)];
            for (int i = 0; i < aa.Length; i++)
                aa[i] = a[i, idx];
            return aa;
        }

        static double[,] _dot(double[,] a, double[,] b)
        {
            Debug.Assert(a.GetLength(1) == b.GetLength(0));
            double[,] dot = new double[a.GetLength(0),b.GetLength(1)];

            Console.WriteLine("a {0}", a2s(_shape(a)));
            Console.WriteLine("b {0}", a2s(_shape(b)));

            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < b.GetLength(1); j++)
                {
//                    Console.WriteLine("dotproduct {0}", _dot1(_row(a, i), _col(b, j)));
                    dot[i, j] = _dot1(_row(a, i), _col(b, j));
                }
            }

            return dot;
        }

        static double[,] _add(double[,] a, double[,] b)
        {
            Debug.Assert(a.GetLength(0) == b.GetLength(0));
            Debug.Assert(a.GetLength(1) == b.GetLength(1));
            double[,] c = new double[a.GetLength(0), a.GetLength(1)];

            for (int i = 0; i < a.GetLength(0); i++)
                for (int j = 0; j < a.GetLength(1); j++)
                    c[i, j] = a[i, j] + b[i, j];
            return c;
        }

        static double[] identity_function(double[] x)
        {
            return x;
        }

        static void Main(string[] args)
        {
            // step function
            {
                int[] y = step_function(new double[] { -1.0, 1.0, 2.0 });
                Console.WriteLine(a2s(y));
            }

            // sigmoid function
            {
                double[] yy = sigmoid(new double[] { -1, 1, 2 });
                Console.WriteLine(a2s(yy));
            }

            // ReLU function
            {
                double[] y3 = relu(new double[] { -1, 1, 2 });
                Console.WriteLine("y3 = " + a2s(y3));
            }

            // _arange() test
            {
                double[] x = _arange(-5.0, 5.0, 0.1);
                Console.WriteLine(a2s(x));
            }

            // round test
            /*
            double v = 0.333333;
            System.Console.WriteLine(v.ToString());
            System.Console.WriteLine(System.Math.Round(v, 5).ToString());
            System.Console.WriteLine(System.Math.Round(v, 4).ToString());
            System.Console.WriteLine(System.Math.Round(v, 3).ToString());
            System.Console.WriteLine(System.Math.Round(v, 2).ToString());
            System.Console.WriteLine(System.Math.Round(v, 1).ToString());
            */

            // 3.3.1 多次元配列
            {
                double[,] a = new double[3, 2];
                Console.WriteLine(a.GetLength(0));
                Console.WriteLine(a.GetLength(1));
                Console.WriteLine("a.shape = {0}", a2s(_shape(a)));
            }

            // 3.3.2 行列の内積
            double[,] dot = _dot(new double[,] { { 1, 2 }, { 3, 4 } }, new double[,] { { 5, 6 }, { 7, 8 } });
            Console.WriteLine("{0}", a2s(dot));
            // must be [[19, 22], [43, 50]]

            dot = _dot(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }, new double[,] { { 7 }, { 8 } });
            Console.WriteLine("{0}", a2s(dot));
            // must be [[23], [53], [83]]

            // 3.3.3 ニューラルネットワークの内積
            {
                double[,] X = new double[,] { { 1, 2 } };
                Console.WriteLine("X.shape {0}", a2s(_shape(X)));
                Console.WriteLine("X {0}", a2s(X));
                double[,] W = new double[,] { { 1, 3, 5 }, { 2, 4, 6 } };
                Console.WriteLine("W.shape {0}", a2s(_shape(W)));
                Console.WriteLine("W {0}", a2s(W));
                double[,] Y = _dot(X, W);
                Console.WriteLine("Y.shape {0}", a2s(_shape(Y)));
                Console.WriteLine("Y {0}", a2s(Y));
            }

            // 3.4 3層ニューラルネットワークの実装
            {
                double[,] X = new double[,] { { 1.0, 0.5 } };
                double[,] W1 = new double[,] { { 0.1, 0.3, 0.5 }, { 0.2, 0.4, 0.6 } };
                double[,] B1 = new double[,] { { 0.1, 0.2, 0.3 } };
                //print(W1.shape) # (2, 3)
                //print(X.shape) # (2,)
                //print(B1.shape) # (3,)
                Console.WriteLine("X.shape {0}", a2s(_shape(X)));
                Console.WriteLine("W1.shape {0}", a2s(_shape(W1)));
                Console.WriteLine("B1.shape {0}", a2s(_shape(B1)));

                double[,] A1 = _add(_dot(X, W1), B1);
                Console.WriteLine("A1 {0}", a2s(A1));
                double[] Z1;
                Z1 = sigmoid(_row(A1, 0));
                Console.WriteLine("Z1 {0}", a2s(Z1));
                // must be [0.57444252, 0.66818777, 0.75026011]

                double[,] _Z1 = new double[1, Z1.Length];
                for (int i = 0; i < _Z1.GetLength(1); i++)
                    _Z1[0, i] = Z1[i];
                Console.WriteLine("_Z1 {0}", a2s(_Z1));

                double[,] W2 = new double[,] { { 0.1, 0.4 }, { 0.2, 0.5 }, { 0.3, 0.6 } };
                double[,] B2 = new double[,] { { 0.1, 0.2 } };

                double[,] A2 = _add(_dot(_Z1, W2), B2);
                double[] Z2 = sigmoid(_row(A2, 0));
                Console.WriteLine("Z2 {0}", a2s(Z2));

                double[,] W3 = new double[,] { { 0.1, 0.3 }, { 0.2, 0.4 } };
                double[,] B3 = new double[,] { { 0.1, 0.2 } };

                double[,] _Z2 = new double[1, Z2.Length];
                for (int i = 0; i < _Z2.GetLength(1); i++)
                    _Z2[0, i] = Z2[i];
                Console.WriteLine("_Z2 {0}", a2s(_Z2));

                double[,] A3 = _add(_dot(_Z2, W3), B3);
                Console.WriteLine("A3 {0}", a2s(A3));
                double[] Y = identity_function(_row(A3, 0));
                Console.WriteLine("Y {0}", a2s(Y));
            }

            // 3.4.3 実装のまとめ
            {
                Hashtable network = init_network();
                double[,] x = new double[,] { { 1.0, 0.5 } };
                double[] y = forward(network, x);
                Console.WriteLine("y {0}", a2s(y));
                // must be [ 0.31682708 0.69627909]
            }
        }

        static Hashtable init_network()
        {
            Hashtable network = new Hashtable();

            double[,] W1 = new double[,] { { 0.1, 0.3, 0.5 }, { 0.2, 0.4, 0.6 } };
            double[,] B1 = new double[,] { { 0.1, 0.2, 0.3 } };
            double[,] W2 = new double[,] { { 0.1, 0.4 }, { 0.2, 0.5 }, { 0.3, 0.6 } };
            double[,] B2 = new double[,] { { 0.1, 0.2 } };
            double[,] W3 = new double[,] { { 0.1, 0.3 }, { 0.2, 0.4 } };
            double[,] B3 = new double[,] { { 0.1, 0.2 } };

            network["W1"] = W1;
            network["b1"] = B1;
            network["W2"] = W2;
            network["b2"] = B2;
            network["W3"] = W3;
            network["b3"] = B3;

            return network;
        }

        static double[] forward(Hashtable network, double[,] x)
        {
            double[,] W1 = (double[,])network["W1"];
            double[,] B1 = (double[,])network["b1"];
            double[,] W2 = (double[,])network["W2"];
            double[,] B2 = (double[,])network["b2"];
            double[,] W3 = (double[,])network["W3"];
            double[,] B3 = (double[,])network["b3"];

            double[,] A1 = _add(_dot(x, W1), B1);
            Console.WriteLine("A1 {0}", a2s(A1));
            double[] Z1 = sigmoid(_row(A1, 0));
            Console.WriteLine("Z1 {0}", a2s(Z1));
            // must be [0.57444252, 0.66818777, 0.75026011]

            double[,] _Z1 = new double[1, Z1.Length];
            for (int i = 0; i < _Z1.GetLength(1); i++)
                _Z1[0, i] = Z1[i];
            Console.WriteLine("_Z1 {0}", a2s(_Z1));

            double[,] A2 = _add(_dot(_Z1, W2), B2);
            double[] Z2 = sigmoid(_row(A2, 0));
            Console.WriteLine("Z2 {0}", a2s(Z2));

            double[,] _Z2 = new double[1, Z2.Length];
            for (int i = 0; i < _Z2.GetLength(1); i++)
                _Z2[0, i] = Z2[i];
            Console.WriteLine("_Z2 {0}", a2s(_Z2));

            double[,] A3 = _add(_dot(_Z2, W3), B3);
            Console.WriteLine("A3 {0}", a2s(A3));
            double[] Y = identity_function(_row(A3, 0));
            Console.WriteLine("Y {0}", a2s(Y));

            return Y;
        }
    }
}
