using System;
using System.Diagnostics;
using System.IO;

using org.snaga.numeric;

namespace Ch4
{
    class simpleNet
    {
        public double[,] W;

        public simpleNet()
        {
            W = np.random_randn(2, 3);
            W = new double[,] { { 0.47355232, 0.9977393, 0.84668094 }, { 0.85557411, 0.03563661, 0.69422093 } };
        }

        public double[,] predict(double[,] x)
        {
            Debug.Assert(x.GetLength(1) == W.GetLength(0));
            return np.dot(x, W);
        }

        public double loss(double[,] x, double[,] t)
        {
            double[,] z = predict(x);
            double[,] y = nn.softmax(z);
            double loss = nn.cross_entropy_error(y, t);
            return loss;
        }

        static void _Main(string[] args)
        {
            simpleNet net = new simpleNet();
            Console.WriteLine(" W = " + np.str(net.W));

            double[] x = new double[] { 0.6, 0.9 };
            Console.WriteLine(" x = " + np.str(x));

            double[,] p = net.predict(np.reshape(x, 1, 2));
            Console.WriteLine(" p = " + np.str(p));
        }
    }

    class TwoLayerNet
    {
        public double[,] W1;
        public double[,] b1;
        public double[,] W2;
        public double[,] b2;

        public TwoLayerNet(int input_size, int hidden_size, int output_size, double weight_init_std = 0.01)
        {
            W1 = np.multi(np.random_randn(input_size, hidden_size), weight_init_std);
            b1 = np.reshape(np.zeros(hidden_size), 1, hidden_size);
            W2 = np.multi(np.random_randn(hidden_size, output_size), weight_init_std);
            b2 = np.reshape(np.zeros(output_size), 1, output_size);

            Console.WriteLine("W1.shape = " + np.str(np.shape(W1)));
            Console.WriteLine("b1.shape = " + np.str(np.shape(b1)));
            Console.WriteLine("W1.shape = " + np.str(np.shape(W2)));
            Console.WriteLine("b2.shape = " + np.str(np.shape(b2)));
        }

        public double[] predict(double[] x)
        {
            double[,] a1;
            double[,] z1;
            double[,] a2;
            double[,] y;
            double[,] xx;

            xx = np.reshape(x, 1, x.Length);

            a1 = np.add(np.dot(xx, W1), b1);
            //                Console.WriteLine("  a1.shape = " + np.str(np.shape(a1)));
            z1 = nn.sigmoid(a1);
            //                Console.WriteLine("  z1.shape = " + np.str(np.shape(z1)));
            a2 = np.add(np.dot(z1, W2), b2);
            //                Console.WriteLine("  a2.shape = " + np.str(np.shape(a2)));
            y = nn.softmax(a2);
            //                Console.WriteLine("  y.shape = " + np.str(np.shape(y)));

            //                Console.WriteLine("y.shape = " + np.str(np.shape(y)));
            Debug.Assert(y.GetLength(0) == 1);
            return np.row(y, 0);
        }

        // バッチ対応
        public double[,] predict(double[,] x)
        {
            double[,] a1;
            double[,] z1;
            double[,] a2;
            double[,] y;

            a1 = np.add(np.dot(x, W1), b1);
            z1 = nn.sigmoid(a1);
            a2 = np.add(np.dot(z1, W2), b2);
            y = nn.softmax(a2);
            return y;
        }

        public double loss(double[] x, double[] t)
        {
            double[] y = predict(x);
            //                Console.WriteLine("loss: t.shape = " + np.str(np.shape(t)));
            //                Console.WriteLine("loss: y.shape = " + np.str(np.shape(y)));
            return nn.cross_entropy_error(y, t);
        }

        // バッチ対応
        public double loss(double[,] x, double[,] t)
        {
            double[,] y = predict(x);
            return nn.cross_entropy_error(y, t);
        }

        public double accuracy(double[] x, double[] t)
        {
            double[] y = predict(x);
            int yy = np.argmax(y);
            int tt = np.argmax(t);
            int count = 0;

            Debug.Assert(y.Length == t.Length);

            for (int i = 0; i < y.Length; i++)
                if (y[i] == t[i])
                    count++;
            return (double)count / (double)y.Length;
        }

        public double[,] numerical_gradient(double[] x, double[] t, ref double[,] p)
        {
            double h = 1e-4;
            double[,] bak;
            double[,] grad;

            bak = p;
            grad = np.zeros_like(p);
            for (int i = 0; i < p.GetLength(0); i++)
            {
                for (int j = 0; j < p.GetLength(1); j++)
                {
                    p[i, j] = bak[i, j] + h;
                    double fxh1 = loss(x, t);
                    p[i, j] = bak[i, j] - h;
                    double fxh2 = loss(x, t);
                    grad[i, j] = (fxh1 - fxh2) / (2 * h);
                    p = bak;
                }
                Console.WriteLine("numerical_gradient: {0}/{1}", i, p.GetLength(0));
            }
            Console.WriteLine("numerical_gradient: grad.shape = {0}", np.str(np.shape(grad)));
            return grad;
        }

        public double[,] numerical_gradient(double[,] x, double[,] t, ref double[,] p)
        {
            double h = 1e-4;
            double bak;
            double[,] grad;

            grad = np.zeros_like(p);
            for (int i = 0; i < p.GetLength(0); i++)
            {
                for (int j = 0; j < p.GetLength(1); j++)
                {
                    bak = p[i, j];
                    p[i, j] = bak + h;
                    double fxh1 = loss(x, t);
                    p[i, j] = bak - h;
                    double fxh2 = loss(x, t);
                    grad[i, j] = (fxh1 - fxh2) / (2 * h);
                    Console.WriteLine("grad[{0},{1}] = {2}", i, j, grad[i, j]);
                    p[i, j] = bak;
                }
                Console.WriteLine("numerical_gradient: {0}/{1}", i, p.GetLength(0));
                return grad;
            }
            Console.WriteLine("numerical_gradient: grad.shape = {0}", np.str(np.shape(grad)));
            return grad;
        }
        static void Main(string[] args)
        {
            double[,] W = np.random_randn(784, 100);
            double[,] x = np.random_randn(100, 784);

            Console.WriteLine(DateTime.Now.ToString("F"));
            for (int i = 0; i < 1000; i++)
            {
                np.dot(W, x);
                //                Console.WriteLine("{0}", i);
            }
            Console.WriteLine(DateTime.Now.ToString("F"));
        }

        static void _Main(string[] args)
        {
            TwoLayerNet network = new TwoLayerNet(784, 100, 10);

            double[,] x = np.random_rand(100, 784); // 100 inputs
            Console.WriteLine(" x.shape = " + np.str(np.shape(x)));

            double[] y = network.predict(np.row(x, 0)); // 1 output
            Console.WriteLine(" y.shape = " + np.str(np.shape(y)));

            double[,] y2 = network.predict(x); // 100 outputs
            Console.WriteLine(" y2.shape = " + np.str(np.shape(y2)));

            double[,] t = np.random_rand(100, 10); // 100 outputs

#if NOT_USED
            Console.WriteLine(DateTime.Now.ToString("F"));
            for (int j = 0; j < 100; j++)
                for (int i = 0; i < t.GetLength(0); i++)
                    network.loss(np.row(x, i), np.row(t, i));

            Console.WriteLine(DateTime.Now.ToString("F"));
            for (int j = 0; j < 100; j++)
                network.loss(x, t);

            Console.WriteLine(DateTime.Now.ToString("F"));
            Environment.Exit(0);
#endif

            double[,] grad = network.numerical_gradient(x, t, ref network.W1);
            /*
                        Console.WriteLine(" grad.shape = " + np.str(np.shape(grad)));
                        Console.WriteLine(" grad = " + np.str(grad));
            */
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            // 4.2.1 2 乗和誤差
            double[] t = { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
            double[] y = { 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0 };

            Console.WriteLine("MSE = {0}", nn.mean_squared_error(y, t));
            Debug.Assert(Math.Round(nn.mean_squared_error(y, t), 4) == 0.0975);

            y = new double[] { 0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0 };
            Console.WriteLine("MSE = {0}", nn.mean_squared_error(y, t));
            Debug.Assert(Math.Round(nn.mean_squared_error(y, t), 4) == 0.5975);

            // 4.2.2 交差エントロピー誤差
            t = new double[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
            y = new double[] { 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0 };
            Console.WriteLine("MSE = {0}", nn.cross_entropy_error(y, t));
            Debug.Assert(Math.Round(nn.cross_entropy_error(y, t), 10) == Math.Round(0.510825457099338, 10));

            y = new double[] { 0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0 };
            Console.WriteLine("MSE = {0}", nn.cross_entropy_error(y, t));
            Debug.Assert(Math.Round(nn.cross_entropy_error(y, t), 10) == Math.Round(2.30258409299455, 10));

            // 4.2.3 ミニバッチ学習
            double[,] x_train = read_data("x_train.csv", 60000, 784);
            double[,] t_train = read_data("t_train_onehot.csv", 60000, 10);

            Console.WriteLine(np.str(np.rows(x_train, 0, 10)));
            Console.WriteLine(np.str(np.rows(t_train, 0, 10)));

            int train_size = (np.shape(x_train))[0];
            int batch_size = 3;
            int[] batch_mask = np.random_choice(train_size, batch_size);
            Console.WriteLine("batch mask {0}", np.str(batch_mask));

            double[,] x_batch = np.rows(x_train, batch_mask);
            Console.WriteLine("x_batch = {0}", np.str(x_batch));

            double[,] t_batch = np.rows(t_train, batch_mask);
            Console.WriteLine("t_batch = {0}", np.str(t_batch));

            // 4.2.4 ［バッチ対応版］交差エントロピー誤差の実装

            // 4.3.2 数値微分の例
            Console.WriteLine(numerical_diff(function_1, 5));
            Debug.Assert(Math.Round(numerical_diff(function_1, 5), 14) == 0.19999999999909);

            Console.WriteLine(numerical_diff(function_1, 10));
            Debug.Assert(Math.Round(numerical_diff(function_1, 10), 15) == 0.299999999998635);

            // 4.3.3 偏微分
            Console.WriteLine(numerical_diff(function_tmp1, 3.0));
            Debug.Assert(Math.Round(numerical_diff(function_tmp1, 3.0), 14) == 6.00000000000378);

            Console.WriteLine(numerical_diff(function_tmp2, 4.0));
            Debug.Assert(Math.Round(numerical_diff(function_tmp2, 4.0), 14) == 7.99999999999912);

            // 4.4 勾配
            Console.WriteLine(np.str(numerical_gradient(function_2, new double[] { 3.0, 4.0 })));
            Debug.Assert(np.str(numerical_gradient(function_2, new double[] { 3.0, 4.0 })) == "[ 6.00000000000378, 7.99999999999912 ]");

            Console.WriteLine(np.str(numerical_gradient(function_2, new double[] { 0.0, 2.0 })));
            Debug.Assert(np.str(numerical_gradient(function_2, new double[] { 0.0, 2.0 })) == "[ 0, 4.000000000004 ]");

            Console.WriteLine(np.str(numerical_gradient(function_2, new double[] { 3.0, 0.0 })));
            Debug.Assert(np.str(numerical_gradient(function_2, new double[] { 3.0, 0.0 })) == "[ 6.00000000001266, 0 ]");

            // 4.4.1 勾配法
            double[] init_x = new double[] { -3.0, 4.0 };
            Console.WriteLine(np.str(gradient_descent(function_2, init_x, 0.1, 100)));
            Debug.Assert(np.str(gradient_descent(function_2, init_x, 0.1, 100)) == "[ -6.11110792899879E-10, 8.14814390531427E-10 ]");

            // 学習率が大きすぎる例：lr=10.0
            Console.WriteLine(np.str(gradient_descent(function_2, init_x, 10.0, 100)));
            Debug.Assert(np.str(gradient_descent(function_2, init_x, 10.0, 100)) == "[ -25898374737328.4, -1295248616896.54 ]");

            // 学習率が小さすぎる例：lr=1e-10
            Console.WriteLine(np.str(gradient_descent(function_2, init_x, 1e-10, 100)));
            Debug.Assert(np.str(gradient_descent(function_2, init_x, 1e-10, 100)) == "[ -2.99999994, 3.99999991999999 ]");

            // 4.4.2 ニューラルネットワークに対する勾配
            simpleNet net = new simpleNet();
            // for debug
            net.W = new double[,] {
                { 0.47355232, 0.9977393, 0.84668094},
                { 0.85557411, 0.03563661, 0.69422093 }
            };
            Console.WriteLine("W = " + np.str(net.W));

            double[,] x = new double[,] { { 0.6, 0.9 } };
            Console.WriteLine("x = " + np.str(x));

            double[,] p = net.predict(x);
            Console.WriteLine("p = " + np.str(p));
            // for debug
            Debug.Assert(np.str(p) == "[ [ 1.054148091, 0.630716529, 1.132807401 ] ]");

            Console.WriteLine("argmax = " + np.argmax(np.row(p, 0)));
            Debug.Assert(np.argmax(np.row(p, 0)) == 2);

            double[,] tt = new double[,] { { 0, 0, 1 } };
            Console.WriteLine("loss = " + net.loss(x, tt));
            // for debug
            Debug.Assert(net.loss(x, tt) == 0.928068538748235);

            // delegate関数から参照するために、クラス変数に保存する。
            Program.xx = x;
            Program.tt = tt;
            Program.net = net;
            double[,] dW = np.zeros_like(net.W);

            double[,] w_bak = net.W;
            dW = numerical_gradient(f, net.W);
            Console.WriteLine("grad = {0}", np.str(dW));

            Debug.Assert(np.str(net.W) == np.str(w_bak));
        }

        public static double[,] xx;
        public static double[,] tt;
        public static simpleNet net;

        public static double f(double[] W)
        {
            Debug.Assert(xx.GetLength(0) == 1 && tt.GetLength(0) == 1);

            return net.loss(xx, tt);
        }

        public static double[] gradient_descent(numerical_diff_func f, double[] init_x, double lr=0.01, int step_num=100)
        {
            double[] x = init_x;
            for (int i = 0; i < step_num; i++)
            {
                double[] grad = numerical_gradient(f, x);
                x = np.subtract(x, np.multi(grad, lr));
            }
            return x;
        }

        // 複数の x を受け取り、それぞれの x に対応する勾配 grad を返却する。
        // ここでは偏微分をするので、xの各要素を個別に取り出して勾配を計算する
        public static double[] numerical_gradient(numerical_diff_func f, double[] x)
        {
            double h = 1e-4;
            double[] grad = new double[x.Length];

            for (int i = 0; i < x.Length; i++)
            {
                double tmp_val = x[i];

                x[i] = tmp_val + h;
                double fxh1 = f(x); // xに対応するyの値(1)
                x[i] = tmp_val - h;
                double fxh2 = f(x); // xに対応するyの値(2)

                grad[i] = (fxh1 - fxh2) / (2*h); // xの前後におけるyの勾配
                x[i] = tmp_val;
            }
            return grad;
        }

        // ------------------------------------------------------
        // p.111 で使う numerical_gradient() 関数の実装。
        //
        // Pythonは参照渡しのため、引数として渡されたxを書き換えると
        // simpleNet クラスの W が書き換えられて損失が計算される。
        //
        // C# は値渡しのため、引数として渡した x を書き換えても simpleNet の W の値は
        // 書き変わらない。そのため、重みのパラメータを偏微分する処理が動作しない。
        // ここでは simpleNet の W を直接書き換えて、偏微分処理後に元に戻す。
        // ------------------------------------------------------
        public static double[,] numerical_gradient(numerical_diff_func f, double[,] x)
        {
            double h = 1e-4;
            double[,] grad = np.zeros_like(x);
            double[,] x_bak = x;

            for (int i = 0; i < x.GetLength(0); i++)
            {
                for (int j = 0; j < x.GetLength(1); j++)
                {
                    double tmp_val = x[i, j];

                    x[i, j] = tmp_val + h;
                    net.W = x;
                    double fxh1 = f(np.row(x, i)); // xに対応するyの値を計算する(1)

                    x[i, j] = tmp_val - h;
                    net.W = x;
                    double fxh2 = f(np.row(x, i)); // xに対応するyの値を計算する(2)

                    grad[i,j] = (fxh1 - fxh2) / (2 * h); // xの前後におけるyの勾配を計算する

                    net.W = x_bak; // 重みの値を元に戻す。
                }
            }
            return grad;
        }

        // 1つ以上の説明変数(x)を受け取って、ひとつの被説明変数(y)を返す関数のdelegate
        public delegate double numerical_diff_func(double[] x);

        // 説明変数が1つの場合
        public static double numerical_diff(numerical_diff_func f, double x)
        {
            return numerical_diff(f, new double[] { x });
        }

        // 説明変数が複数の場合
        public static double numerical_diff(numerical_diff_func f, double[] x)
        {
            double h = 1e-4;
            return (f(np.add(x, h)) - f(np.add(x, -h))) / (h * 2);
        }
        
        public static double function_1(double[] x)
        {
            return 0.01 * Math.Pow(x[0], 2) + 0.1 * x[0];
        }

        public static double function_2(double[] x)
        {
            return Math.Pow(x[0], 2) + Math.Pow(x[1], 2);
        }

        public static double function_tmp1(double[] x)
        {
            return x[0] * x[0] + Math.Pow(4, 2);
        }

        public static double function_tmp2(double[] x)
        {
            return Math.Pow(3, 2) + x[0] * x[0];
        }

        static double[,] read_data(string file, int rows, int cols)
        {
            var csv = new CsvHelper.CsvReader(new StreamReader(file));
            csv.Configuration.HasHeaderRecord = false;
            double[,] data = new double[rows, cols];

            Console.WriteLine("Reading {0}... rows={1}, cols={2}", file, rows, cols);
            for (int i = 0; i < rows; i++)
            {
                if (!csv.Read())
                    break;
                if (i % 1000 == 0)
                    Console.WriteLine("Reading {0}... {1}", file, i);
                for (int j = 0; j < cols; j++)
                {
                    try
                    {
                        data[i, j] = Double.Parse(csv.GetField(j));
                    }
                    catch (CsvHelper.CsvMissingFieldException e)
                    {
                        Console.WriteLine("read_data: " + np.str(np.row(data, i)));
                        break;
                    }
                }
            }
            Console.WriteLine("Reading {0}... done.", file);
            return data;
        }

    }
}
