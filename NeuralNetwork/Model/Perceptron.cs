using System;
using NeuralNetwork.Interface;

namespace NeuralNetwork.Model
{
    public class Perceptron : AbstractNeuralNetwork
    {
		public Perceptron(int inputCount)
		{
			weights = new double[inputCount];
			RandomInit(); // random seeding init val for weights and bias
		}

        /// <summary>
        /// The perceptron is trained to respond to each input vector with a corresponding target output of either 0 or 1. 
        /// The learning rule has been proven to converge on a solution in finite time if a solution exists.
        /// 
        /// The learning rule can be summarized in the following two equations:
        /// 
        /// b = b + [ T - A ]
        /// 
        /// For all inputs i:
        /// 
        /// W(i) = W(i) + [ T - A ] * P(i)
        /// 
        /// Where W is the vector of weights, 
        /// P is the input vector presented to the network, 
        /// T is the correct result that the neuron should have shown, 
        /// A is the actual output of the neuron, and b is the bias.
        /// </summary>
        /// <returns>true if converge; false if not</returns>
        public override bool LearningRule(double[,] inputs, double[] outputs)
		{
			if (inputs.GetLength(0) != outputs.Length)
				throw new InvalidOperationException();
			if (inputs.GetLength(1) != weights.Length)
				throw new InvalidOperationException();

            if (activationFunc == null)
            {
                throw new ArgumentException("Haven't set activation function");
            }

            bool notPassed = true;
			int count = 0;
			while(notPassed)
			{
				bool allPass = true;
				double adjustment = 0.0;
				for (int i = 0; i < inputs.GetLength(0); i++)
				{
					double calcOutput = activationFunc.ProcessValue(GetSum(GetRow(inputs, i)));
					adjustment = outputs[i] - calcOutput;

					bias += adjustment; // b = b + [ T - A ]
                    allPass &= adjustment.Equals(0.0);

					for (int j = 0; j < weights.Length; j++)
					{
						weights[j] += (adjustment * inputs[i, j]); // W(i) = W(i) + [ T - A ] * P(i)
                    }
				}
				notPassed = !allPass;
				count++;
				if (count > MAX) return false;
			} 
			return true;
		}

        public override double TrainingOutput(double[] input)
        {
            if (activationFunc == null)
            {
                throw new ArgumentException("Haven't set activation function");
            }

            return activationFunc.ProcessValue(GetSum(input));
        }


        private void RandomInit()
        {
            Random rnd = new Random();
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = rnd.NextDouble();
            }
            bias = rnd.NextDouble();
        }

        private T[] GetRow<T>(T[,] matrix, int rowNum)
		{
			if (rowNum >= matrix.GetLength(0)) return null;

			T[] array = new T[matrix.GetLength(1)];
			for(int i = 0; i < matrix.GetLength(1); i++)
			{
				array[i] = matrix[rowNum, i];
			}
			return array;
		}


    }
}
