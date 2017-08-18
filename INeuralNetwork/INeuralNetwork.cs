using System;

namespace NeuralNetworkInterface
{
    public interface INeuralNetwork
    {
		bool LearningRule(double[,] inputs, double[] outputs);
		double ActivationFunc(double sum);
        double Summation(double[] inputs);


        double TrainingOutput(double[] input);
    }
}
