using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

using org.snaga.numeric;

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

        static double[,] sigmoid(double[,] x)
        {
            double[,] ss = new double[x.GetLength(0), x.GetLength(1)];

            for (int i = 0; i < x.GetLength(0); i++)
            {
                double[] s = sigmoid(np.row(x, i));
                np.row(ss, i, s);
            }
            return ss;
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
                Console.WriteLine("a.shape = {0}", a2s(np.shape(a)));
            }

            // 3.3.2 行列の内積
            {
                double[,] dot = np.dot(new double[,] { { 1, 2 }, { 3, 4 } }, new double[,] { { 5, 6 }, { 7, 8 } });
                Console.WriteLine("{0}", a2s(dot));
                // must be [[19, 22], [43, 50]]

                dot = np.dot(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }, new double[,] { { 7 }, { 8 } });
                Console.WriteLine("{0}", a2s(dot));
                // must be [[23], [53], [83]]
            }

            // 3.3.3 ニューラルネットワークの内積
            {
                double[,] X = new double[,] { { 1, 2 } };
                Console.WriteLine("X.shape {0}", a2s(np.shape(X)));
                Console.WriteLine("X {0}", a2s(X));
                double[,] W = new double[,] { { 1, 3, 5 }, { 2, 4, 6 } };
                Console.WriteLine("W.shape {0}", a2s(np.shape(W)));
                Console.WriteLine("W {0}", a2s(W));
                double[,] Y = np.dot(X, W);
                Console.WriteLine("Y.shape {0}", a2s(np.shape(Y)));
                Console.WriteLine("Y {0}", a2s(Y));
            }

            // 3.4 3層ニューラルネットワークの実装
            {
                Console.WriteLine("3.4 3層ニューラルネットワークの実装: start");

                double[,] X = new double[,] { { 1.0, 0.5 } };
                double[,] W1 = new double[,] { { 0.1, 0.3, 0.5 }, { 0.2, 0.4, 0.6 } };
                double[,] B1 = new double[,] { { 0.1, 0.2, 0.3 } };
                //print(W1.shape) # (2, 3)
                //print(X.shape) # (2,)
                //print(B1.shape) # (3,)
                Console.WriteLine("X.shape {0}", a2s(np.shape(X)));
                Console.WriteLine("W1.shape {0}", a2s(np.shape(W1)));
                Console.WriteLine("B1.shape {0}", a2s(np.shape(B1)));

                double[,] A1 = np.add(np.dot(X, W1), B1);
                Console.WriteLine("A1 {0}", a2s(A1));
                double[] Z1;
                Z1 = sigmoid(np.row(A1, 0));
                Console.WriteLine("Z1 {0}", a2s(Z1));
                // must be [0.57444252, 0.66818777, 0.75026011]

                double[,] _Z1 = np._2D(Z1);
                Console.WriteLine("_Z1 {0}", a2s(_Z1));

                double[,] W2 = new double[,] { { 0.1, 0.4 }, { 0.2, 0.5 }, { 0.3, 0.6 } };
                double[,] B2 = new double[,] { { 0.1, 0.2 } };

                double[,] A2 = np.add(np.dot(_Z1, W2), B2);
                double[] Z2 = sigmoid(np.row(A2, 0));
                Console.WriteLine("Z2 {0}", a2s(Z2));

                double[,] W3 = new double[,] { { 0.1, 0.3 }, { 0.2, 0.4 } };
                double[,] B3 = new double[,] { { 0.1, 0.2 } };

                double[,] _Z2 = np._2D(Z2);
                Console.WriteLine("_Z2 {0}", a2s(_Z2));

                double[,] A3 = np.add(np.dot(_Z2, W3), B3);
                Console.WriteLine("A3 {0}", a2s(A3));
                double[] Y = identity_function(np.row(A3, 0));
                Console.WriteLine("Y {0}", a2s(Y));

                Console.WriteLine("3.4 3層ニューラルネットワークの実装: end");
            }

            // 3.4.3 実装のまとめ
            {
                Console.WriteLine("3.4.3 実装のまとめ: start");

                Hashtable network = init_network();
                double[,] x = new double[,] { { 1.0, 0.5 } };
                double[] y = forward(network, x);
                Console.WriteLine("y {0}", a2s(y));
                // must be [ 0.31682708 0.69627909]

                Console.WriteLine("3.4.3 実装のまとめ: end");
            }

            // 3.5 出力層の設計
            {
                Console.WriteLine("3.5 出力層の設計: start");

                double[] a = { 0.3, 2.9, 4.0 };
                double[] b = np.exp(a);
                Console.WriteLine("b {0}", a2s(b));
                Console.WriteLine("b.sum {0}", np.sum(b));
                Console.WriteLine("b/b.sum {0}", a2s(np.div(b, np.sum(b))));
                Console.WriteLine("softmax(a) {0}", a2s(softmax(a)));
                // must be [ 0.01821127 0.24519181 0.73659691]
                Console.WriteLine("softmax(a).sum {0}", np.sum(softmax(a)));

                Console.WriteLine("3.5 出力層の設計: end");
            }

            // 3.6 手書き数字認識
            {
                Console.WriteLine("3.6 手書き数字認識: start");

                double[,] x;
                double[,] t;
                get_data(out x, out t);

                Console.WriteLine("x = " + a2s(np.row(x, 0)));
                Console.WriteLine("t = " + a2s(np.row(t, 0)));

                Hashtable network = init_network2();

                int accuracy_cnt = 0;
                for (int j = 0; j < x.GetLength(0); j++)
                {
                    double[] xx = np.row(x, j);
#if _DEBUG
                    Console.WriteLine("xx = {0}", a2s(xx));
#endif
                    double[] y = predict(network, np._2D(xx));
                    int idx = np.argmax(y);
                    Console.WriteLine("{0}: idx = {1}, prob = {2}, t = {3}", j, idx, y[idx], t[j, 0]);
                    if (idx == t[j, 0])
                        accuracy_cnt += 1;

                    Console.WriteLine("{0}: accuracy: {1}", j, (double)accuracy_cnt / (j + 1));
                }
            }

            // 3.6.3 バッチ処理
            {
                Console.WriteLine("3.6.3 バッチ処理: start");

                double[,] x;
                double[,] t;
                get_data(out x, out t);

                Console.WriteLine("x = " + a2s(np.row(x, 0)));
                Console.WriteLine("t = " + a2s(np.row(t, 0)));

                Hashtable network = init_network2();

                int accuracy_cnt = 0;

                double[,] y = predict_batch(network, x);

                for (int j = 0; j < y.GetLength(0); j++)
                {
                    int idx = np.argmax(np.row(y, j));
                    if (idx == t[j, 0])
                        accuracy_cnt += 1;

                    if (j % 100 == 0)
                        Console.WriteLine("{0}: accuracy: {1}", j, (double)accuracy_cnt / (j + 1));
                }
                Console.WriteLine("accuracy: {0}", (double)accuracy_cnt / y.GetLength(0));

                Console.WriteLine("3.6.3 バッチ処理: end");
            }
        }

        static double[] predict(Hashtable network, double[,] x)
        {
            double[] y;

            double[,] W1 = (double[,])network["W1"];
            double[,] W2 = (double[,])network["W2"];
            double[,] W3 = (double[,])network["W3"];
            double[,] b1 = (double[,])network["b1"];
            double[,] b2 = (double[,])network["b2"];
            double[,] b3 = (double[,])network["b3"];

            // 入力層
#if _DEBUG
            Console.WriteLine("x.shape = {0}", a2s(np.shape(x)));
            Console.WriteLine("W1.shape = {0}", a2s(np.shape(W1)));
            Console.WriteLine("b1 = {0}", a2s(np.shape(b1)));
            Console.WriteLine("x dot W1 = {0}", a2s(np.shape(np.dot(x, W1))));
#endif

            double[,] a1 = np.add(np.dot(x, W1), b1);
#if _DEBUG
            Console.WriteLine("a1.shape = {0}", a2s(np.shape(a1)));
            Console.WriteLine("a1 = {0}", a2s(a1));
#endif
            double[,] z1 = np._2D(sigmoid(np.row(a1, 0)));
#if _DEBUG
            Console.WriteLine("z1.shape = {0}", a2s(np.shape(z1)));
            Console.WriteLine("z1 = {0}", a2s(z1));
#endif

            // 隠れ層
#if _DEBUG
            Console.WriteLine("W2.shape = {0}", a2s(np.shape(W2)));
            Console.WriteLine("b2.shape = {0}", a2s(np.shape(b2)));
#endif

            double[,] a2 = np.add(np.dot(z1, W2), b2);
#if _DEBUG
            Console.WriteLine("a2.shape = {0}", a2s(np.shape(a2)));
            Console.WriteLine("a2 = {0}", a2s(a2));
#endif

            double[,] z2 = np._2D(sigmoid(np.row(a2, 0)));
#if _DEBUG
            Console.WriteLine("z2.shape = {0}", a2s(np.shape(z2)));
            Console.WriteLine("z2 = {0}", a2s(z2));
#endif

            // 出力層
#if _DEBUG
            Console.WriteLine("W3.shape = {0}", a2s(np.shape(W3)));
            Console.WriteLine("b3.shape = {0}", a2s(np.shape(b3)));
#endif

            double[,] a3 = np.add(np.dot(z2, W3), b3);
#if _DEBUG
            Console.WriteLine("a3.shape = {0}", a2s(np.shape(a3)));
            Console.WriteLine("a3 = {0}", a2s(a3));
#endif
            y = softmax(np.row(a3, 0));

#if _DEBUG
            Console.WriteLine("y = {0}", a2s(y));
#endif
            return y;
        }

        static double[,] predict_batch(Hashtable network, double[,] x)
        {
            double[,] y;

            double[,] W1 = (double[,])network["W1"];
            double[,] W2 = (double[,])network["W2"];
            double[,] W3 = (double[,])network["W3"];
            double[,] b1 = (double[,])network["b1"];
            double[,] b2 = (double[,])network["b2"];
            double[,] b3 = (double[,])network["b3"];

            // 入力層
            Console.WriteLine("x.shape = {0}", a2s(np.shape(x)));
            Console.WriteLine("W1.shape = {0}", a2s(np.shape(W1)));
            Console.WriteLine("b1.shape = {0}", a2s(np.shape(b1)));
            Console.WriteLine("x dot W1 = {0}", a2s(np.shape(np.dot(x, W1))));

            double[,] a1 = np.add(np.dot(x, W1), b1);

            Console.WriteLine("a1.shape = {0}", a2s(np.shape(a1)));
//            Console.WriteLine("a1 = {0}", a2s(a1));

            double[,] z1 = sigmoid(a1);

            Console.WriteLine("z1.shape = {0}", a2s(np.shape(z1)));
//            Console.WriteLine("z1 = {0}", a2s(z1));

            // 隠れ層
            Console.WriteLine("W2.shape = {0}", a2s(np.shape(W2)));
            Console.WriteLine("b2.shape = {0}", a2s(np.shape(b2)));

            double[,] a2 = np.add(np.dot(z1, W2), b2);

            Console.WriteLine("a2.shape = {0}", a2s(np.shape(a2)));
//            Console.WriteLine("a2 = {0}", a2s(a2));

            double[,] z2 = sigmoid(a2);

            Console.WriteLine("z2.shape = {0}", a2s(np.shape(z2)));
//            Console.WriteLine("z2 = {0}", a2s(z2));

            // 出力層
            Console.WriteLine("W3.shape = {0}", a2s(np.shape(W3)));
            Console.WriteLine("b3.shape = {0}", a2s(np.shape(b3)));

            double[,] a3 = np.add(np.dot(z2, W3), b3);

            Console.WriteLine("a3.shape = {0}", a2s(np.shape(a3)));
//            Console.WriteLine("a3 = {0}", a2s(a3));

            y = softmax(a3);

//            Console.WriteLine("y = {0}", a2s(y));
            Console.WriteLine("y.shape = {0}", a2s(np.shape(y)));

            return y;
        }

        static void get_data(out double[,] x_test, out double[,] t_test)
        {
            t_test = read_data("t_test.csv", 10000, 1);
            x_test = read_data("x_test.csv", 10000, 784);
        }

        static Hashtable init_network2()
        {
            Hashtable network = new Hashtable();

            double[,] W1 = read_data("sample_weight_W1.csv", 784, 50);
            double[,] W2 = read_data("sample_weight_W2.csv", 50, 100);
            double[,] W3 = read_data("sample_weight_W3.csv", 100, 10);
            double[,] b1 = read_data("sample_weight_b1.csv", 1, 50);
            double[,] b2 = read_data("sample_weight_b2.csv", 1, 100);
            double[,] b3 = read_data("sample_weight_b3.csv", 1, 10);

            network["W1"] = W1;
            network["b1"] = b1;
            network["W2"] = W2;
            network["b2"] = b2;
            network["W3"] = W3;
            network["b3"] = b3;

            return network;
        }

        static double[,] read_data(string file, int rows, int cols)
        {
            var csv = new CsvHelper.CsvReader(new StreamReader(file));
            csv.Configuration.HasHeaderRecord = false;
            double[,] data = new double[rows,cols];

            Console.WriteLine("Reading {0}... rows={1}, cols={2}", file, rows, cols);
            for (int i = 0; i < rows; i++)
            {
                if (!csv.Read())
                    break;
                if (i % 100 == 0)
                    Console.WriteLine("Reading {0}... {1}", file, i);
                for (int j = 0; j < cols; j++)
                {
                    try
                    {
                        data[i, j] = Double.Parse(csv.GetField(j));
                    }
                    catch (CsvHelper.CsvMissingFieldException e)
                    {
                        Console.WriteLine("read_data: " + a2s(np.row(data, i)));
                        break;
                    }
                }
            }
            Console.WriteLine("Reading {0}... done.", file);
            return data;
        }

        static double[] softmax(double[] a)
        {
            double c = np.max(a);
            double[] exp_a = np.exp(np.add(a, -1.0 * c));
            double[] y = np.div(exp_a, np.sum(exp_a));
            return y;
        }

        static double[,] softmax(double[,] a)
        {
            double[,] ss = new double[a.GetLength(0), a.GetLength(1)];

            for (int i = 0; i < a.GetLength(0); i++)
            {
                double[] s = softmax(np.row(a, i));
                np.row(ss, i, s);
            }
            return ss;
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

            double[,] A1 = np.add(np.dot(x, W1), B1);
            Console.WriteLine("A1 {0}", a2s(A1));
            double[] Z1 = sigmoid(np.row(A1, 0));
            Console.WriteLine("Z1 {0}", a2s(Z1));
            // must be [0.57444252, 0.66818777, 0.75026011]

            double[,] _Z1 = new double[1, Z1.Length];
            for (int i = 0; i < _Z1.GetLength(1); i++)
                _Z1[0, i] = Z1[i];
            Console.WriteLine("_Z1 {0}", a2s(_Z1));

            double[,] A2 = np.add(np.dot(_Z1, W2), B2);
            double[] Z2 = sigmoid(np.row(A2, 0));
            Console.WriteLine("Z2 {0}", a2s(Z2));

            double[,] _Z2 = new double[1, Z2.Length];
            for (int i = 0; i < _Z2.GetLength(1); i++)
                _Z2[0, i] = Z2[i];
            Console.WriteLine("_Z2 {0}", a2s(_Z2));

            double[,] A3 = np.add(np.dot(_Z2, W3), B3);
            Console.WriteLine("A3 {0}", a2s(A3));
            double[] Y = identity_function(np.row(A3, 0));
            Console.WriteLine("Y {0}", a2s(Y));

            return Y;
        }
    }
}
