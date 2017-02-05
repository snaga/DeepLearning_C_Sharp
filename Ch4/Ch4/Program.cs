using System;
using System.Diagnostics;
using System.IO;

using org.snaga.numeric;

namespace Ch4
{
    class Program
    {
        static double mean_squared_error(double[] y, double[] t)
        {
            return 0.5 * np.sum(np.pow(np.subtract(y, t), 2));
        }

        static double cross_entropy_error(double[] y, double[] t)
        {
            double delta = 1e-7;
            return -1.0 * np.sum(np.multi(t, np.log(np.add(y, delta))));
        }

        static double[] cross_entropy_error(double[,] y, double[,] t)
        {
            Debug.Assert(y.GetLength(0) == t.GetLength(0));
            Debug.Assert(y.GetLength(1) == t.GetLength(1));

            int batch_size = y.GetLength(0);

            return np.multi(np.div(np.sum(np.multi(t, np.log(y))), batch_size), -1);
        }

        static void Main(string[] args)
        {
            // 4.2.1 2 乗和誤差
            double[] t = { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
            double[] y = { 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0 };

            Console.WriteLine("MSE = {0}", mean_squared_error(y, t));
            Debug.Assert(Math.Round(mean_squared_error(y, t), 4) == 0.0975);

            y = new double[] { 0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0 };
            Console.WriteLine("MSE = {0}", mean_squared_error(y, t));
            Debug.Assert(Math.Round(mean_squared_error(y, t), 4) == 0.5975);

            // 4.2.2 交差エントロピー誤差
            t = new double[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
            y = new double[] { 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0 };
            Console.WriteLine("MSE = {0}", cross_entropy_error(y, t));
            Debug.Assert(Math.Round(cross_entropy_error(y, t), 10) == Math.Round(0.510825457099338, 10));

            y = new double[] { 0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0 };
            Console.WriteLine("MSE = {0}", cross_entropy_error(y, t));
            Debug.Assert(Math.Round(cross_entropy_error(y, t), 10) == Math.Round(2.30258409299455, 10));

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
            Debug.Assert(Math.Round(numerical_diff(function_1, 5),14) == 0.19999999999909);

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

        public static double[] numerical_gradient(numerical_diff_func f, double[] x)
        {
            double h = 1e-4;
            double[] grad = new double[x.Length];

            for (int i = 0; i < x.Length; i++)
            {
                double tmp_val = x[i];

                x[i] = tmp_val + h;
                double fxh1 = f(x);
                x[i] = tmp_val - h;
                double fxh2 = f(x);

                grad[i] = (fxh1 - fxh2) / (2*h);
                x[i] = tmp_val;
            }
            return grad;
        }

        public delegate double numerical_diff_func(double[] x);

        public static double numerical_diff(numerical_diff_func f, double x)
        {
            return numerical_diff(f, new double[] { x });
        }

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
