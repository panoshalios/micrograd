import random
from itertools import chain
from micrograd.value import Value


class Module:

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.gradient = 0.0

    def parameters(self) -> list[Value]:
        return []


class Neuron(Module):
    # Contains weights and biases
    def __init__(self, input_dim, non_linear=True) -> None:
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(input_dim)]
        self.bias = Value(0) # Should this not be initialized randomly
        self.non_linear = non_linear

    def __call__(self, input):
        matrix_mult = (wi * xi for wi, xi in zip(self.weights, input))
        activations = sum(matrix_mult, self.bias) # Is the bias added properly here

        # Add non - linearity here or after the output
        return activations.relu() if self.non_linear else activations

    def parameters(self) -> list[Value]:
        return self.weights + [self.bias]
    

class Layer(Module):
    def __init__(self, input_dim, output_dim, **kwargs) -> None:
        self.neurons = [Neuron(input_dim, **kwargs) for _ in range(output_dim)]

    def __call__(self, input):
        out = [neuron(input) for neuron in self.neurons]
        return out

    def parameters(self) -> list[Value]:
        param_list = map(Neuron.parameters, self.neurons)
        return list(chain.from_iterable(param_list))


class MLP(Module):
    def __init__(self, input_dimensionality: int, hidden_layer_dimensions: list[int]):
        all_dims = [input_dimensionality] + hidden_layer_dimensions
        self.layers = [
            Layer(all_dims[i], all_dims[i + 1], non_linear=i != len(hidden_layer_dimensions) - 1)
            for i in range(len(all_dims) - 1)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Value]:
        params = map(Layer.parameters, self.layers)
        return list(chain.from_iterable(params))