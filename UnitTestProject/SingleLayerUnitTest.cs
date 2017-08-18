using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Model;

namespace UnitTestProject
{
	[TestClass]
	public class SingleLayerUnitTest
	{
		[TestMethod]
		public void ANDLogic()
		{
			Perceptron AddGate = new Perceptron(2);
            AddGate.SetActivationFunc(new SimpleActivation());

            double[,] inputs = new double[4, 2]
			{
				{ 0, 0 },
				{ 0, 1 },
				{ 1, 0 },
				{ 1, 1 }
			};
			double[] outputs = new double[4] { 0, 0, 0, 1 };
			Assert.IsTrue(AddGate.LearningRule(inputs, outputs)); // can converge

            Assert.IsTrue(AddGate.TrainingOutput(new double[] { 0, 0 }).Equals(0));
            Assert.IsTrue(AddGate.TrainingOutput(new double[] { 0, 1 }).Equals(0));
            Assert.IsTrue(AddGate.TrainingOutput(new double[] { 1, 0 }).Equals(0));
            Assert.IsTrue(AddGate.TrainingOutput(new double[] { 1, 1 }).Equals(1));
        }

		[TestMethod]
		public void ORLogic()
		{
			Perceptron OrGate = new Perceptron(2);
            OrGate.SetActivationFunc(new SimpleActivation());

            double[,] inputs = new double[4, 2]
			{
				{ 0, 0 },
				{ 0, 1 },
				{ 1, 0 },
				{ 1, 1 }
			};
			double[] outputs = new double[4] { 0, 1, 1, 1 };
			Assert.IsTrue(OrGate.LearningRule(inputs, outputs)); // can converge

            Assert.IsTrue(OrGate.TrainingOutput(new double[] { 0, 0 }).Equals(0));
            Assert.IsTrue(OrGate.TrainingOutput(new double[] { 0, 1 }).Equals(1));
            Assert.IsTrue(OrGate.TrainingOutput(new double[] { 1, 0 }).Equals(1));
            Assert.IsTrue(OrGate.TrainingOutput(new double[] { 1, 1 }).Equals(1));
        }

		[TestMethod]
		public void XORLogic()
		{
			Perceptron XORGate = new Perceptron(2);
            XORGate.SetActivationFunc(new SimpleActivation());

            double[,] inputs = new double[4, 2]
			{
				{ 0, 0 },
				{ 0, 1 },
				{ 1, 0 },
				{ 1, 1 }
			};
			double[] outputs = new double[4] { 0, 1, 1, 0 };
			// single layer cannot represent XOR logic
			Assert.IsFalse(XORGate.LearningRule(inputs, outputs)); // cannot converge
        }

	}
}
