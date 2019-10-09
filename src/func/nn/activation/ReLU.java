package func.nn.activation;

/**
 * RELU activation function
 * #this is simple, RELU is a flat line when x < 0, and a line of slope 1 when x > 0
 * @author Sean Lam slam36@gatech.edu
 * @version 1.0
 */
public class ReLU extends DifferentiableActivationFunction {

    /**
     * @see nn.function.DifferentiableActivationFunction#derivative(double)
     */
    public double derivative(double value) {
        if (value < 0) {
            return 0;
        } else {
            return 1;
        }
    }

    /**
     * @see nn.function.ActivationFunction#activation(double)
     */
    public double value(double value) {
        if (value < 0) {
            return 0;
        } else {
            return value;
        }
    }

}
