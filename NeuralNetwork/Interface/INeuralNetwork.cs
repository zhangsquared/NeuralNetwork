using System;

namespace NeuralNetwork.Interface
{
    public interface INeuralNetwork
    {
		bool LearningRule(double[,] inputs, double[] outputs);

        double TrainingOutput(double[] input);

        void SetActivationFunc(IActivation func);

        double GetSum(double[] inputs);
    }
}
