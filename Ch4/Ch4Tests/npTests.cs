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
        public void reshapeTest()
        {
            double[] a = { 1, 2, 3, 4, 5, 6 };
            double[,] b = { { 1, 2, 3 }, { 4, 5, 6 } };
            double[,] c = { { 1, 2, 3, 4, 5, 6 } };
            double[,] d;

            d = np.reshape(a, 2, 3);
            CollectionAssert.AreEqual(b, d);

            d = np.reshape(a, 1, 6);
            CollectionAssert.AreEqual(c, d);
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

        [TestMethod()]
        public void addTest2()
        {
            double[] a = new double[] { 1, 2, 3 };
            double[] b = new double[] { 5, 6, 7 };
            double[] c = new double[] { 6, 8, 10 };
            double[] d;

            d = np.add(a, b);

            CollectionAssert.AreEqual(c, d);
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
        public void powTest()
        {
            double[] a = new double[] { 1, 2, 3 };
            double[] b = new double[] { 1, 4, 9 };
            double[] c;

            c = np.pow(a, 2);

            CollectionAssert.AreEqual(b, c);
        }

        [TestMethod()]
        public void logTest()
        {
            double[] a = new double[] { 1, 2, 3 };
            double[] b = new double[] { 0, 0.6931471805599453, 1.09861228866810969 };
            double[] c;

            c = np.log(a);

            CollectionAssert.AreEqual(b, c);
        }

        [TestMethod()]
        public void multiTest()
        {
            double[] a = new double[] { 1, 2, 3 };
            double[] b = new double[] { 4, 5, 6 };
            double[] c = new double[] { 4, 10, 18 };
            double[] d;

            d = np.multi(a, b);

            CollectionAssert.AreEqual(c, d);
        }

        [TestMethod()]
        public void random_choiceTest()
        {
            int[] b = np.random_choice(10, 3);

            Assert.IsTrue(3 == b.Length);
            Assert.IsTrue(b[0] >= 0 && b[0] <= 10);
            Assert.IsTrue(b[1] >= 0 && b[1] <= 10);
            Assert.IsTrue(b[2] >= 0 && b[2] <= 10);
        }

        [TestMethod()]
        public void rowsTest()
        {
            double[,] a = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 10, 11, 12 } };
            int[] b = { 0, 2 };
            double[,] c = { { 1, 2, 3 }, { 7, 8, 9 } };

            double[,] d = np.rows(a, b);

            CollectionAssert.AreEqual(c, d);
        }

        [TestMethod()]
        public void addTest3()
        {
            double[,] a = { { 1, 2, 3 }, { 4, 5, 6 } };
            double b = 7;
            double[,] c = { { 8, 9, 10 }, { 11, 12, 13 } };
            double[,] d;

            d = np.add(a, b);

            CollectionAssert.AreEqual(c, d);
        }

        [TestMethod()]
        public void logTest1()
        {
            double[,] a = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            double[,] b = new double[,] { { 0, 0.6931471805599453, 1.09861228866810969 }, { 1.3862943611198906, 1.6094379124341003, 1.791759469228055 } };
            double[,] c;

            c = np.log(a);

            CollectionAssert.AreEqual(b, c);
        }

        [TestMethod()]
        public void multiTest1()
        {
            double[,] a = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            double[,] b = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            double[,] c = new double[,] { { 1, 4, 9 }, { 16, 25, 36 } };

            double[,] d = np.multi(a, b);

            CollectionAssert.AreEqual(c, d);
        }

        [TestMethod()]
        public void sumTest1()
        {
            double[,] a = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            double[] b = { 6, 15 };

            double[] c = np.sum(a);

            CollectionAssert.AreEqual(b, c);
        }

        [TestMethod()]
        public void multiTest2()
        {
            double[] a = new double[] { 1, 2, 3 };
            double[] b = new double[] { 2, 4, 6 };

            double[] c = np.multi(a, 2);

            CollectionAssert.AreEqual(b, c);
        }
    }
}