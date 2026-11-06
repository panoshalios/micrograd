
from __future__ import annotations

import math


class Value:
    def __init__(self, data: float, children=(), operation='', label=''):
        self.label = label
        self.operation = operation
        self.children = set(children)
        # Gradient of dfinal/dself
        self.gradient = 0.0
        self._backward = lambda: None
        self.data = data

    def __add__(self, other) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        result_data = self.data + other.data
        children = (self, other)
        result = Value(result_data, children=children, operation="+")

        def add_backward():
            self.gradient += result.gradient
            other.gradient += result.gradient

        result._backward = add_backward
        return result

    def __mul__(self, other) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        result_data = self.data * other.data
        children = (self, other)
        result = Value(result_data, children=children, operation="*")

        def mul_backward():
            self.gradient += other.data * result.gradient
            other.gradient += self.data * result.gradient

        result._backward = mul_backward
        return result

    def __pow__(self, other) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        result_data = self.data ** other.data
        children = (self,)
        result = Value(result_data, children=children, operation=f"**{other}")

        def pow_backward():
            self.gradient += (other * self.data**(other-1)) * result.gradient

        result._backward = pow_backward
        return result

    def relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')

        def _backward():
            self.gradient += (out.data > 0) * out.gradient
        out._backward = _backward

        return out

    def log(self):
        result = Value(math.log(self.data), (self,), 'log')

        def log_backward():
            self.gradient += (1 / self.data) * result.gradient
        
        result._backward = log_backward
        return result

    def sigmoid(self):
        # Keep exp(-x) from overflowing when x is negative, and keep the probability strictly inside (0, 1) so log is defined.
        if self.data >= 0:
            exp_term = math.exp(-self.data)
            result_data = 1 / (1 + exp_term)
        else:
            exp_term = math.exp(self.data)
            result_data = exp_term / (1 + exp_term)
        eps = 1e-12
        result_data = min(max(result_data, eps), 1 - eps)
        result = Value(result_data, (self,), 'sigmoid')

        def sigmoid_backward():
            self.gradient += result.data * (1 - result.data) * result.gradient

        result._backward = sigmoid_backward
        return result

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: Value) -> Value:
        return self + (-other)

    def __radd__(self, other: Value) -> Value:
        return self + other

    def __rsub__(self, other: Value) -> Value:
        return other + (-self)

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, gradient={self.gradient})"

    def backward(self):
        """Does a backward pass through all the values"""

        # Build the topological graph iteratively so large batches don't hit Python's recursion limit
        graph = []
        visited = set()
        stack = [(self, False)]

        while stack:
            node, processed = stack.pop()
            if processed:
                graph.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for child in node.children:
                if child not in visited:
                    stack.append((child, False))

        self.gradient = 1
        # Starting from the final output run a backward pass
        for node in reversed(graph):
            node._backward()
