using System;

namespace NeuralNetwork.Interface
{
    public abstract class AbstractNeuralNetwork
    {
        protected IActivation activationFunc;

        protected double[] weights;
        protected double bias;
        protected int MAX = 1000000;


        public abstract bool Learn(double[,] inputs, double[] outputs);

        public abstract double TrainingOutput(double[] input);


        public virtual void SetActivationFunc(IActivation func)
        {
            activationFunc = func;
        }

        public double GetSum(double[] inputs)
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

        public void UnLearn()
        {
            RandomInit();
        }

        /// <summary>
        /// random seeding init val for weights and bias
        /// </summary>
        protected void RandomInit()
        {
            Random rnd = new Random();
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = rnd.NextDouble();
            }
            bias = rnd.NextDouble();
        }

    }

}
