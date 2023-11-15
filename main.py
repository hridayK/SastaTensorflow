import numpy as np
from sasta_tensorflow import layers,activations,loss

input = [1,2,3,4,115]

dense1 = layers.Dense(5,5)
act1 = activations.softmax()
dense1.forward(input=input)
act1.forward(dense1.output)
l = loss.categorical_cross_entropy()
print(act1.output)
