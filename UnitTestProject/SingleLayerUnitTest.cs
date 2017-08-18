using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkModel;

namespace UnitTestProject
{
	[TestClass]
	public class SingleLayerUnitTest
	{
		[TestMethod]
		public void ANDLogic()
		{
			Perceptron AddGate = new Perceptron(2);

			double[,] inputs = new double[4, 2]
			{
				{ 0, 0 },
				{ 0, 1 },
				{ 1, 0 },
				{ 1, 1 }
			};
			double[] outputs = new double[4] { 0, 0, 0, 1 };
			Assert.IsTrue(AddGate.Train(inputs, outputs));

            Assert.IsTrue(AddGate.TrainingOutput(new double[] { 0, 0 }).Equals(0));
            Assert.IsTrue(AddGate.TrainingOutput(new double[] { 0, 1 }).Equals(0));
            Assert.IsTrue(AddGate.TrainingOutput(new double[] { 1, 0 }).Equals(0));
            Assert.IsTrue(AddGate.TrainingOutput(new double[] { 1, 1 }).Equals(1));
        }

		[TestMethod]
		public void ORLogic()
		{
			Perceptron OrGate = new Perceptron(2);

			double[,] inputs = new double[4, 2]
			{
				{ 0, 0 },
				{ 0, 1 },
				{ 1, 0 },
				{ 1, 1 }
			};
			double[] outputs = new double[4] { 0, 1, 1, 1 };
			Assert.IsTrue(OrGate.Train(inputs, outputs));

            Assert.IsTrue(OrGate.TrainingOutput(new double[] { 0, 0 }).Equals(0));
            Assert.IsTrue(OrGate.TrainingOutput(new double[] { 0, 1 }).Equals(1));
            Assert.IsTrue(OrGate.TrainingOutput(new double[] { 1, 0 }).Equals(1));
            Assert.IsTrue(OrGate.TrainingOutput(new double[] { 1, 1 }).Equals(1));
        }

		[TestMethod]
		public void XORLogic()
		{
			Perceptron perceptron = new Perceptron(2);

			double[,] inputs = new double[4, 2]
			{
				{ 0, 0 },
				{ 0, 1 },
				{ 1, 0 },
				{ 1, 1 }
			};
			double[] outputs = new double[4] { 0, 1, 1, 0 };
			// single layer cannot represent XOR logic
			Assert.IsFalse(perceptron.Train(inputs, outputs));
		}

	}
}
