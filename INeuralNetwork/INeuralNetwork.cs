using System;

namespace NeuralNetworkInterface
{
    public interface INeuralNetwork
    {
		bool Train(double[,] inputs, double[] outputs);
		double ActivationFunc(double sum);
        double Summation(double[] inputs);


        double TrainingOutput(double[] input);
    }
}
