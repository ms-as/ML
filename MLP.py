import numpy as np
from random import random



class MLP:

    def __init__(self, num_inputs, num_hidden, num_outputs): #constructor
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [num_inputs] + num_hidden + [num_outputs] #its a list, which each element represents number of neurons in the layer


        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations


        derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives



    def forward_propagate(self, inputs):

        activations = inputs
        self.activations[0] = activations

        for i, w in enumerate(self.weights):

            net_inputs = np.dot(activations, w) #calculate net inputs

            activations = self._sigmoid(net_inputs) #calculate activations

            self.activations[i+1] = activations



        return activations

    def _sigmoid(self, x):
        y = 1.0 / (1.0 +np.exp(-x))
        return y
    
    def back_propagate(self, error, verbose = False):

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0],-1)

            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print("derivatives for W{}: {}".format(i, self.derivatives))

        return error

    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate

    def train(self, inputs, targets, lups, learning_rate):
        for i in range(lups):
            sum_errors = 0

            for j, input in enumerate(inputs):
                target = targets[j]

                output = self.forward_propagate(input)

                error = target - output
                self.back_propagate(error)
                self.gradient_descent(learning_rate)

                sum_errors += self._mse(target, output)

            print("Error: {} at loop {}".format(sum_errors / len(items), i+1))

        print("Training complete!")
        print("=====")



    def _mse(self, target, output):
        return np.average((target - output)**2)

            



    def _sigmoid_derivative(self,x):
        return x * (1.0 - x)

    
if __name__ == "__main__":
    

    items = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in items])
    mlp = MLP(2,[5],1)
    mlp.train(items, targets, 150, 0.2)
    input = np.array([0.6, 0.3])
    target = np.array([0.9])
    output = mlp.forward_propagate(input)

    print()
    print("That artificial network find that {} + {} is near to {}".format(input[0], input[1], output[0]))