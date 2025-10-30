import numpy as np
import pickle


class NeuralNetwork:
    def __init__(self, number_of_inputs, number_of_hidden_nodes, number_of_hidden_layers, number_of_outputs, learning_rate):
        self.number_of_inputs = number_of_inputs
        self.number_of_hidden_nodes = number_of_hidden_nodes
        self.number_of_hidden_layers = number_of_hidden_layers
        self.number_of_outputs = number_of_outputs
        self.learning_rate = learning_rate

        self.input_to_hidden_weights = np.random.randn(number_of_hidden_nodes,number_of_inputs) * np.sqrt(2 / number_of_inputs)

        self.hidden_weights = np.random.randn(number_of_hidden_layers-1,number_of_hidden_nodes,number_of_hidden_nodes) * np.sqrt(2 / number_of_inputs)
        self.hidden_biases = np.zeros((number_of_hidden_layers,number_of_hidden_nodes))

        self.hidden_to_output_weights = np.random.randn(number_of_outputs,number_of_hidden_nodes) * np.sqrt(2 / number_of_inputs)
        self.output_biases = np.zeros((number_of_outputs))


    def activation(self, x):
        return np.maximum(0, x)
        #return 1/(1+np.exp(-x))

    def activation_derivative(self, x):
        return(x>0).astype(float)
        #return x*(1-x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # numerical stability
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


    def feedforward(self,input_layer):
        self.input_layer = np.array(input_layer)
        self.first_hidden_layer = self.activation(np.tensordot(input_layer, self.input_to_hidden_weights, axes=(-1,1)) + self.hidden_biases[0])
        self.hidden_layers = np.repeat(self.first_hidden_layer[np.newaxis], self.number_of_hidden_layers, axis=0)
        for i in range(1,self.number_of_hidden_layers):
            self.hidden_layers[i] = self.activation(np.tensordot(self.hidden_layers[i-1], self.hidden_weights[i-1], axes=(-1,1)) + self.hidden_biases[i])
        self.output_layer = self.softmax(np.tensordot(self.hidden_layers[-1], self.hidden_to_output_weights, axes=(-1,1)))
        return self.output_layer


    def backpropagate(self,output_layer, desired_output_layer):

        output_delta = self.output_layer - desired_output_layer

        hidden_delta = np.zeros_like(self.hidden_layers)

        hidden_delta[-1] = self.activation_derivative(self.hidden_layers[-1]) * np.tensordot(output_delta, self.hidden_to_output_weights, axes=(1, 0))

        for i in range(self.number_of_hidden_layers-2,-1,-1):
            hidden_delta[i] = self.activation_derivative(self.hidden_layers[i]) * np.tensordot(hidden_delta[i+1], self.hidden_weights[i], axes=(1, 0))


        self.hidden_to_output_weights = self.hidden_to_output_weights - self.learning_rate * np.tensordot(output_delta.T,self.hidden_layers[-1],axes = (1,0))
        for i in range(0,self.number_of_hidden_layers-1):
            self.hidden_weights[i] = self.hidden_weights[i] - self.learning_rate * np.tensordot(hidden_delta[i+1].T,self.hidden_layers[i],axes=(1,0))
        self.input_to_hidden_weights = self.input_to_hidden_weights - self.learning_rate * np.tensordot(hidden_delta[0].T,self.input_layer,axes=(1,0))

        self.output_biases -= self.learning_rate * np.sum(output_delta, axis=0)
        self.hidden_biases -= self.learning_rate * np.sum(hidden_delta, axis=1)


    def train(self, epoch,batch_size, data_batch_input, data_batch_expected_output):
        number_of_correct = 0
        print_frequency = 2
        for i in range(epoch):
            for j in range(int(data_batch_input.shape[0]/batch_size)):
                start = ((j)*batch_size)%len(data_batch_input)
                end = min(start + batch_size,len(data_batch_input)-1)
                data_batch_output = self.feedforward(data_batch_input[start:end])
                self.backpropagate(data_batch_output, data_batch_expected_output[start:end])
                correctly_found = np.argmax(data_batch_output, axis=-1) == np.argmax(data_batch_expected_output[start:end],axis=-1)
                number_of_correct += np.sum(correctly_found)
            if i % print_frequency == 0:
                print(f"Epoch {i}, accuracy:{number_of_correct/(data_batch_input.shape[0]*print_frequency)*100:.2f}%")
                number_of_correct = 0

    def save_model(self,file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)

    def load_model(self,file_name):
        with open(file_name, 'rb') as file:
            self = pickle.load(file)