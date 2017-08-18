using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Model
{
    public class SimpleActivation : IActivation
    {
        public double Activate(double sum)
        {
            return sum < 0.0 ? 0.0 : 1.0;
        }
    }
}
