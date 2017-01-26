using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ch2
{
    class Program
    {
        // old function
        static int AND1(int x1, int x2)
        {
            double w1 = 0.5;
            double w2 = 0.5;
            double theta = 0.7;
            double tmp = x1 * w1 + x2 * w2;
            if (tmp <= theta)
                return 0;
            return 1;
        }

        // element-wise product
        static double[] _ewp(int[] a, double[] b)
        {
            double[] c = new double[a.Length];
            for (int i=0; i < a.Length; i++)
                c[i] = a[i] * b[i];
            return c;
        }

        static double _sum(double[] a)
        {
            double b = 0;
            for (int i = 0; i < a.Length; i++)
                b += a[i];
            return b;
        }

        static int AND(int x1, int x2)
        {
            int[] x = new int[] { x1, x2 };
            double[] w = new double[] { 0.5, 0.5 };
            double b = -0.7;

            double tmp = _sum(_ewp(x, w)) + b;
            if (tmp <= 0)
                return 0;
            return 1;
        }

        static int NAND(int x1, int x2)
        {
            int[] x = new int[] { x1, x2 };
            double[] w = new double[] { -0.5, -0.5 };
            double b = 0.7;

            double tmp = _sum(_ewp(x, w)) + b;
            if (tmp <= 0)
                return 0;
            return 1;
        }

        static int OR(int x1, int x2)
        {
            int[] x = new int[] { x1, x2 };
            double[] w = new double[] { 0.5, 0.5 };
            double b = -0.2;

            double tmp = _sum(_ewp(x, w)) + b;
            if (tmp <= 0)
                return 0;
            else
                return 1;
        }

        static int XOR(int x1, int x2)
        {
            int s1 = NAND(x1, x2);
            int s2 = OR(x1, x2);
            int y = AND(s1, s2);
            return y;
        }

        static void Main(string[] args)
        {
            Console.WriteLine("Hello, Deep Learning World.");

            Console.WriteLine("AND(0, 0) => {0}", AND(0, 0));
            Console.WriteLine("AND(1, 0) => {0}", AND(1, 0));
            Console.WriteLine("AND(0, 1) => {0}", AND(0, 1));
            Console.WriteLine("AND(1, 1) => {0}", AND(1, 1));

            Console.WriteLine("NAND(0, 0) => {0}", NAND(0, 0));
            Console.WriteLine("NAND(1, 0) => {0}", NAND(1, 0));
            Console.WriteLine("NAND(0, 1) => {0}", NAND(0, 1));
            Console.WriteLine("NAND(1, 1) => {0}", NAND(1, 1));

            Console.WriteLine("OR(0, 0) => {0}", OR(0, 0));
            Console.WriteLine("OR(1, 0) => {0}", OR(1, 0));
            Console.WriteLine("OR(0, 1) => {0}", OR(0, 1));
            Console.WriteLine("OR(1, 1) => {0}", OR(1, 1));

            Console.WriteLine("XOR(0, 0) => {0}", XOR(0, 0));
            Console.WriteLine("XOR(1, 0) => {0}", XOR(1, 0));
            Console.WriteLine("XOR(0, 1) => {0}", XOR(0, 1));
            Console.WriteLine("XOR(1, 1) => {0}", XOR(1, 1));
        }
    }
}
