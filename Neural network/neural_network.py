from math import sqrt
from random import uniform

from numpy import dot, transpose
from scipy.special import expit


class Neural_network:
    def __init__(self, learning_rate, num_hidden):
        self.L = learning_rate

        self.w_i_h = self.random_weights(num_hidden, 784)
        self.w_h_o = self.random_weights(10, num_hidden)

    def random_weights(self, height, width):
        min_value = 0
        max_value = 1 / sqrt(width)

        matrix = [[] for _ in range(height)]
        for i in range(height):
            for _ in range(width):
                matrix[i].append(uniform(min_value, max_value))

        return matrix

    def backpropagate(self, inputs, target_outputs):
        predicted = self.predict(inputs)

        error_output = target_outputs - predicted
        error_hidden = dot(transpose(self.w_h_o), error_output)

        return error_output, error_hidden

    def train_network(self, inputs, target_outputs):
        inputs = [[i] for i in inputs]
        target_outputs = [[i] for i in target_outputs]

        hidden_inputs = dot(self.w_i_h, inputs)
        hidden_outputs = sigmoid(hidden_inputs)

        output_inputs = dot(self.w_h_o, hidden_outputs)
        output_outputs = sigmoid(output_inputs)

        error_output, error_hidden = self.backpropagate(inputs, target_outputs)

        self.w_h_o += self.L * dot((error_output * output_outputs * (1 - output_outputs)), transpose(hidden_outputs))
        self.w_i_h += self.L * dot((error_hidden * hidden_outputs * (1 - hidden_outputs)), transpose(inputs))

    def predict(self, inputs):
        #inputs must be 2D array

        hidden_outputs = dot(self.w_i_h, inputs)
        hidden_outputs = sigmoid(hidden_outputs)

        outputs = dot(self.w_h_o, hidden_outputs)
        outputs = sigmoid(outputs)

        return outputs


sigmoid = lambda x: expit(x)