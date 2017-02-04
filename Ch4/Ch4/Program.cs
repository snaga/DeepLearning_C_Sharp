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
