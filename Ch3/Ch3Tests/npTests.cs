using Microsoft.VisualStudio.TestTools.UnitTesting;
using org.snaga.numeric;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

namespace org.snaga.numeric.Tests
{
    [TestClass()]
    public class npTests
    {
        [TestMethod()]
        public void shapeTest()
        {
            double[,] a = new double[2, 3];
            int[] b = new int[] { 2, 3 };
            CollectionAssert.AreEqual(b, np.shape(a));
        }

        [TestMethod()]
        public void addTest()
        {
            Assert.Fail();
        }

        [TestMethod()]
        public void addTest1()
        {
            Assert.Fail();
        }

        [TestMethod()]
        public void divTest()
        {
            Assert.Fail();
        }

        [TestMethod()]
        public void maxTest()
        {
            Assert.Fail();
        }

        [TestMethod()]
        public void sumTest()
        {
            double[] a = { 0.1, 0.02, 0.003 };
            Assert.AreEqual(Math.Round(np.sum(a), 3), Math.Round(0.123, 3));
        }

        [TestMethod()]
        public void expTest()
        {
            Assert.Fail();
        }

        [TestMethod()]
        public void dotTest()
        {
            Assert.Fail();
        }

        [TestMethod()]
        public void rowTest()
        {
            Assert.Fail();
        }

        [TestMethod()]
        public void colTest()
        {
            Assert.Fail();
        }

        [TestMethod()]
        public void dotTest1()
        {
            double[,] a = new double[,] { { 1, 2 }, { 3, 4 } };
            double[,] b = new double[,] { { 5, 6 }, { 7, 8 } };
            double[,] c = new double[,] { { 19, 22 }, { 43, 50 } };
            double[,] d = np.dot(a, b);

            CollectionAssert.AreEqual(new int[] { 2, 2 }, np.shape(d));
            CollectionAssert.AreEqual(c, d);

            a = new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } };
            b = new double[,] { { 7 }, { 8 } };
            c = new double[,] { { 23 }, { 53 }, { 83 } };
            d = np.dot(a, b);

            CollectionAssert.AreEqual(new int[] { 3, 1 }, np.shape(d));
            CollectionAssert.AreEqual(c, d);
        }

        [TestMethod()]
        public void TTest()
        {
            double[,] a = new double[,] { { 1, 2 }, { 3, 4 } };
            double[,] b = new double[,] { { 1, 3 }, { 2, 4 } };

            CollectionAssert.AreEqual(b, np.T(a));

            a = new double[,] { { 1, 2, 3, 4 } };
            b = new double[,] { { 1 }, { 2 }, { 3 }, { 4 } };

            CollectionAssert.AreEqual(b, np.T(a));
        }

        [TestMethod()]
        public void _2DTest()
        {
            double[] a = new double[] { 1, 2, 3, 4 };
            double[,] b = new double[,] { { 1, 2, 3, 4 } };

            CollectionAssert.AreEqual(b, np._2D(a));
        }

        [TestMethod()]
        public void argmaxTest()
        {
            double[] a = new double[] { 1, 2, 3, 4 };
            Assert.AreEqual(3, np.argmax(a));

            a = new double[] { 1, 7, 3, 4 };
            Assert.AreEqual(1, np.argmax(a));

            a = new double[] { 11, 7, 3, 4 };
            Assert.AreEqual(0, np.argmax(a));

            a = new double[] { 11, 7, 3, 11 };
            Assert.AreEqual(0, np.argmax(a)); // first one wins
        }

        [TestMethod()]
        public void rowTest1()
        {
            double[,] a = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            double[] b = new double[] { 7, 8, 9 };
            double[,] c = new double[,] { { 7, 8, 9 }, { 4, 5, 6 } };
            double[,] d;

            d = np.row(a, 0, b);

            CollectionAssert.AreEqual(c, d);
        }
    }
}