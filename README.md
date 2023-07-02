# SastaTensorflow
An attempt to make a neural networks from scratch

### Present state of the code:
```python
import numpy as np
from sasta_tensorflow import layers

input = [1,2,3,4,5]

dense1 = layers.Dense(5,2)
dense1.forward(input=input)

print(dense1.output)
```
