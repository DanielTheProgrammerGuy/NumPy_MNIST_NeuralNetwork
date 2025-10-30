from NeuralNetwork import NeuralNetwork
import numpy as np

from tensorflow.keras.datasets import mnist
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

nn = NeuralNetwork(28*28,128,7,10,0.004)

#MNIST test set
(training_inputs, training_labels), (test_inputs, test_labels) = mnist.load_data()

training_inputs = training_inputs.reshape(60000,28*28)/255
training_outputs = np.zeros((training_labels.shape[0],10))
for i in range(training_labels.shape[0]):
    training_outputs[i,training_labels[i]] = 1

test_inputs = test_inputs.reshape(10000,28*28)/255
test_labels = test_labels

#training
nn.train(50,32,training_inputs, training_outputs)

#testing
result = nn.feedforward(test_inputs)
numbers_guessed_right = sum(np.argmax(result, axis=-1) == test_labels)
percentage_right = numbers_guessed_right / len(result)

#final result
print(f"{percentage_right*100:.2f}%")

if input("Do you want to save the model? (y/n): ").lower() == "y":
    nn.save_model("neural_network.pkl")