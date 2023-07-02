import numpy as np
class Dense:

    def __init__(self, n_inputs:int, n_neurons:int):
        """
        # Dense Layer
        Parameters:
        - n_inputs: The number of inputs given to the Dense layer
        - n_neurons: The number of neurons one wishes to have in the Dense layer.
        """
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self, input):
        """
        # Forward Propagation
        Parameter:
        - input: must give the inputs in form of an array or a numpy array
        """
        if(type(input)!=np.ndarray):
            input = np.array(input)
        self.output = np.dot(input, self.weights) + self.biases