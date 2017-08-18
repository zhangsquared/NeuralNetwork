using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Interface;
using NeuralNetwork.Model;

namespace UnitTestProject
{
    [TestClass]
    public class XORUnitTest
    {
        private AbstractNeuralNetwork perceptron;

        public XORUnitTest()
        {
            perceptron = new MultiLayerNeuralNetwork();
            perceptron.SetActivationFunc(new SigmoidActivation());
        }

        [TestMethod]
        public void XORLogic()
        {
        }
    }
}
