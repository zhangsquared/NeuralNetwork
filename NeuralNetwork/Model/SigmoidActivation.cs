using System;

namespace NeuralNetwork.Model
{
    public class SigmoidActivation : IActivation
    {
        /// <summary>
        /// The stronger the input, the faster the neuron fires (the higher the firing rates). 
        /// The sigmoid is also very useful in multi-layer networks, 
        /// as the sigmoid curve allows for differentation (which is required in Back Propogation training of multi layer networks).
        /// </summary>
        /// <param name="sum"></param>
        /// <returns></returns>
        public double ProcessValue(double sum)
        {
            return 1.0 / (1.0 + Math.Pow(Math.E, -sum));
        }
    }
}
