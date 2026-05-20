import math

def sigmoid(x):
    return 1/(1+math.exp(-x))

def neuron(weights, inputs, bias):
    z = sum((w * i for w,i in zip(weights, inputs)))
    z += bias
    return sigmoid(z)

#test
inputs = [1.0, 2.0, 3.0]
weights = [2,0.55,-3.4]
bias = 5.4
output = neuron(weights, inputs, bias)
print(f"output = {output}")