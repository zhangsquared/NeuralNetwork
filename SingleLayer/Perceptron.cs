using System;
using NeuralNetworkInterface;

namespace NeuralNetworkModel
{
    public class Perceptron : INeuralNetwork
    {
		private double[] weights;
		private double bias;
		private int MAX = 1000000;

		public Perceptron(int inputCount)
		{
			weights = new double[inputCount];
			RandomInit();
		}

        public double Summation(double[] inputs)
        {
            if (inputs.Length != weights.Length)
                throw new InvalidOperationException();

            double sum = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                sum += (inputs[i] * weights[i]);
            }
            sum += bias;
            return sum;
        }

        public double ActivationFunc(double sum)
        {
            return sum < 0.0 ? 0.0 : 1.0;
        }

        public bool Train(double[,] inputs, double[] outputs)
		{
			if (inputs.GetLength(0) != outputs.Length)
				throw new InvalidOperationException();
			if (inputs.GetLength(1) != weights.Length)
				throw new InvalidOperationException();

			bool notPassed = true;
			int count = 0;
			while(notPassed)
			{
				bool allPass = true;
				double adjustment = 0.0;
				for (int i = 0; i < inputs.GetLength(0); i++)
				{
					double calcOutput = ActivationFunc(Summation(GetRow(inputs, i)));
					adjustment = outputs[i] - calcOutput;

					bias += adjustment;
					allPass &= adjustment.Equals(0.0);

					for (int j = 0; j < weights.Length; j++)
					{
						weights[j] += (adjustment * inputs[i, j]);
					}
				}
				notPassed = !allPass;
				count++;
				if (count > MAX) return false;
			} 
			return true;
		}

        public double TrainingOutput(double[] input)
        {
            return ActivationFunc(Summation(input));
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
