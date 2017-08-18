using System;

namespace NeuralNetwork.Interface
{
    public abstract class AbstractNeuralNetwork
    {
        protected IActivation activationFunc;

        protected double[] weights;
        protected double bias;
        protected int MAX = 1000000;


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

        public abstract bool LearningRule(double[,] inputs, double[] outputs);

        public abstract double TrainingOutput(double[] input);

    }

}
