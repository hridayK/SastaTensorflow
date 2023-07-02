# SastaTensorflow
An attempt to make a neural networks from scratch

### Present state of the code:
```python
import numpy as np
from sasta_tensorflow import layers,activations

input = [1,2,3,4,115]

dense1 = layers.Dense(5,5)
act1 = activations.softmax()
dense1.forward(input=input)
act1.forward(dense1.output)

print(act1.output)

```
