using Microsoft.VisualStudio.TestTools.UnitTesting;
using org.snaga.numeric;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace org.snaga.numeric.Tests
{
    [TestClass()]
    public class nnTests
    {
        [TestMethod()]
        public void sigmoidTest()
        {
            double[] a = new double[] { -1.0, 1.0, 2.0 };

            double[] b = nn.sigmoid(a);

            Assert.AreEqual("[ 0.268941421369995, 0.731058578630005, 0.880797077977882 ]", np.str(b));
        }

        [TestMethod()]
        public void sigmoidTest1()
        {
            // バッチ対応版
            double[,] a = new double[,] { { -1.0, 1.0, 2.0 }, { -1.0, 1.0, 1.0 } };

            double[,] b = nn.sigmoid(a);

            Console.WriteLine(np.str(b));
            Assert.AreEqual("[ [ 0.268941421369995, 0.731058578630005, 0.880797077977882 ],\n  [ 0.268941421369995, 0.731058578630005, 0.731058578630005 ] ]", np.str(b));
        }

        [TestMethod()]
        public void softmaxTest()
        {
            double[] a = { 0.3, 2.9, 4.0 };

            double[] b = nn.softmax(a);

            Assert.AreEqual("[ 0.0182112732955475, 0.245191812935074, 0.736596913769379 ]", np.str(b));
        }

        [TestMethod()]
        public void softmaxTest1()
        {
            // バッチ対応版
            double[,] a = { { 0.3, 2.9, 4.0 }, { 0.3, 2.9, 1.0 } };

            double[,] b = nn.softmax(a);
            Console.WriteLine(np.str(b));

            Assert.AreEqual("[ [ 0.0182112732955475, 0.245191812935074, 0.736596913769379 ],\n  [ 0.0606888521819899, 0.817098807423252, 0.122212340394758 ] ]", np.str(b));
        }

        [TestMethod()]
        public void subtractTest()
        {
            double[] a = new double[] { 1, 2, 3 };
            double[] b = new double[] { 5, 6, 7 };
            double[] c = new double[] { -4, -4, -4 };
            double[] d;

            d = np.subtract(a, b);

            CollectionAssert.AreEqual(c, d);
        }

        [TestMethod()]
        public void mean_squared_errorTest()
        {
            double[] t = { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
            double[] y = { 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0 };

            Console.WriteLine("MSE = {0}", nn.mean_squared_error(y, t));
            Assert.AreEqual(Math.Round(nn.mean_squared_error(y, t), 4), 0.0975);
//            Assert.Fail();
        }

        [TestMethod()]
        public void cross_entropy_errorTest()
        {
            double[] t = new double[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
            double[] y = new double[] { 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0 };

            Assert.AreEqual(Math.Round(0.510825457099338, 10), Math.Round(nn.cross_entropy_error(y, t), 10));
        }

        [TestMethod()]
        public void cross_entropy_errorTest1()
        {
            double[,] t = new double[,] { { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 } };
            double[,] y = new double[,] { { 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0 }, { 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0 } };

//            Console.Write(nn.cross_entropy_error(y, t));

            Assert.AreEqual(Math.Round(8.31446055402883, 9), Math.Round(nn.cross_entropy_error(y, t), 9));

//            Assert.Fail();
        }
    }
}