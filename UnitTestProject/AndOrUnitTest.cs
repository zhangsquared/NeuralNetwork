using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Interface;
using NeuralNetwork.Model;

namespace UnitTestProject
{
    [TestClass]
	public class AndOrUnitTest
	{
        private AbstractNeuralNetwork perceptron;

        public AndOrUnitTest()
        {
            perceptron = new Perceptron(2);
            perceptron.SetActivationFunc(new StepActivation());
        }


		[TestMethod]
		public void ANDLogic()
		{
            perceptron.UnLearn();

            double[,] inputs = new double[4, 2]
			{
				{ 0, 0 },
				{ 0, 1 },
				{ 1, 0 },
				{ 1, 1 }
			};
			double[] outputs = new double[4] { 0, 0, 0, 1 };
			Assert.IsTrue(perceptron.Learn(inputs, outputs)); // can converge

            Assert.IsTrue(perceptron.TrainingOutput(new double[] { 0, 0 }).Equals(0));
            Assert.IsTrue(perceptron.TrainingOutput(new double[] { 0, 1 }).Equals(0));
            Assert.IsTrue(perceptron.TrainingOutput(new double[] { 1, 0 }).Equals(0));
            Assert.IsTrue(perceptron.TrainingOutput(new double[] { 1, 1 }).Equals(1));
        }

		[TestMethod]
		public void ORLogic()
		{
            perceptron.UnLearn();

            double[,] inputs = new double[4, 2]
			{
				{ 0, 0 },
				{ 0, 1 },
				{ 1, 0 },
				{ 1, 1 }
			};
			double[] outputs = new double[4] { 0, 1, 1, 1 };
			Assert.IsTrue(perceptron.Learn(inputs, outputs)); // can converge

            Assert.IsTrue(perceptron.TrainingOutput(new double[] { 0, 0 }).Equals(0));
            Assert.IsTrue(perceptron.TrainingOutput(new double[] { 0, 1 }).Equals(1));
            Assert.IsTrue(perceptron.TrainingOutput(new double[] { 1, 0 }).Equals(1));
            Assert.IsTrue(perceptron.TrainingOutput(new double[] { 1, 1 }).Equals(1));
        }

		[TestMethod]
		public void XORLogic()
		{
            perceptron.UnLearn();

            double[,] inputs = new double[4, 2]
			{
				{ 0, 0 },
				{ 0, 1 },
				{ 1, 0 },
				{ 1, 1 }
			};
			double[] outputs = new double[4] { 0, 1, 1, 0 };
			// single layer cannot represent XOR logic
			Assert.IsFalse(perceptron.Learn(inputs, outputs)); // cannot converge
        }

	}
}
