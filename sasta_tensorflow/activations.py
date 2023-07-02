import numpy as np

class relu:
    """
    # ReLU activation function
    Acts as the ReLU layer.
    ReLU stands for Rectified Linear Unit which is an activation function
    ### Working:
    ```
    if input < 0:
        return 0
    else: 
        return input
    ```
    """

    def forward(self, input):
        self.output = np.maximum(0, input)
 
class softmax:
    """
    # Softmax activation function
    Acts as the softmax layer.
    Converts the input into a probability distribution.
    Due to this nature, this activation functions is widely used for
    multi-class classification.
    """

    def forward(self, input):
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
