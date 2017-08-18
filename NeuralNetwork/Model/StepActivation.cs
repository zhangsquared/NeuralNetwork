namespace NeuralNetwork.Model
{
    public class StepActivation : IActivation
    {
        /// <summary>
        /// A basic on/off type function
        /// </summary>
        /// <param name="sum"></param>
        public double ProcessValue(double sum)
        {
            return sum < 0.0 ? 0.0 : 1.0;
        }
    }
}
