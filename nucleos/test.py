import nucleos
import layers
import activations
import numpy as np
import colors
import neuralpy

def output(s=""):
    print colors.green + str(s) + colors.end


net = nucleos.Network([2, 3, 1])

training_set = [
        (np.array([[1],[1]]), np.array([[1]])),
        (np.array([[1],[0]]), np.array([[0]])),
        (np.array([[0],[1]]), np.array([[0]])),
        (np.array([[0],[0]]), np.array([[0]]))
        ]

epochs = 30
alpha = 1


net2 = neuralpy.Network(2, 3, 1)
net2.biases[0] = np.array(list(net.start.next_.b))
net2.biases[1] = np.array(list(net.end.b))

net2.weights[0] = np.array(list(net.start.next_.w))
net2.weights[1] = np.array(list(net.end.w))



net.train(training_set[:], epochs, alpha)
output(net.forward(training_set[0][0]))
output()
output("now neuralpy")
net2.train(training_set[:], epochs, alpha)
output(net2.feedforward(training_set[0][0]))


