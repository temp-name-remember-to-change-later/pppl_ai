import numpy as npy
import scipy.special
import matplotlib.pyplot as mplt


class NeuralNet:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate

        self.wih = npy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)) #generates random array with values between 0 and 1, distributed normally
        self.who = npy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.activate = lambda x: scipy.special.expit(x) #expit(x) is equal to e^x/(1 + e^x)
        pass


        '''
        Runs a single training session on the network
        inputs_list is the list of inputs, in this case the pixels in the file
        targets_list is the list of targets for each set of inputs, which is what the picture is supposed to show.
        '''
    def train(self, inputs_list, targets_list):
        inputs = npy.array(inputs_list, ndmin=2).T
        targets = npy.array(targets_list, ndmin=2).T

        hidden_inputs = npy.dot(self.wih, inputs)
        hidden_outputs = self.activate(hidden_inputs)

        final_inputs = npy.dot(self.who, hidden_outputs)
        final_outputs = self.activate(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = npy.dot(self.who.T, output_errors)

        self.who += self.lr * npy.dot(output_errors * final_outputs * (1 - final_outputs), npy.transpose(hidden_outputs))
        self.wih += self.lr * npy.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), npy.transpose(inputs))
        pass

        '''
        This takes a file as input, and goes through each entry and trains the network.
        '''
    def trainFromFile(self, filename):
        data_file = open(filename, 'r', encoding='utf-8-sig')
        data_list = data_file.readlines()
        data_file.close()

        print('Training:')
        counter = 0
        length = len(data_list)
        for entry in data_list:
            counter += 1
            print(counter, '/', length)
            all_values = entry.split(',')
            inputs = (npy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = npy.zeros(oun) + 0.01
            targets[int(entry[0])] = 0.99
            self.train(inputs, targets)

        print('Training complete!\n')

    '''
    This takes a set of inputs as an input, then returns what the network "thinks" the output should be.
    '''
    def query(self, inputs_list):
        inputs = npy.array(inputs_list, ndmin=2).T

        hidden_inputs = npy.dot(self.wih, inputs)
        hidden_outputs = self.activate(hidden_inputs)

        final_inputs = npy.dot(self.who, hidden_outputs)
        final_outputs = self.activate(final_inputs)

        return final_outputs


'''
Returns the index of the greatest element in arr
'''
def find(arr):
    currentMax = 0
    for i in range(len(arr)):
        if arr[i] > arr[currentMax]:
            currentMax = i
    return currentMax

inn = 28 * 28 # input nodes; one for each pixel in the 28x28 image
hin = 100     # hidden nodes
oun = 10      # output nodes; one for each possible output, in this case each digit from 0 to 9
lr = 0.2      # learning rate

net = NeuralNet(inn, hin, oun, lr)

net.trainFromFile('mnist_train.csv')

test_data_file = open('mnist_test.csv', 'r')
test_list = test_data_file.readlines()
test_data_file.close()

errors = [] # Every time the network returns the wrong answer, it adds the actual answer to the list of errors
for entry in test_list:
    all_values = entry.split(',')
    target = int(entry[0])
    print('Target:', target, end='; ')
    mplt.show(mplt.imshow(npy.asfarray(all_values[1:]).reshape(28, 28), cmap='Greys', interpolation='None')) # Unomment this line to go through the test examples one by one
    result = net.query(npy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    print("Result =", find(result))
    if target != find(result):
        errors.append(target)

print()
print('List of errors:', errors)
print()
print("Final score:", len(test_list) - len(errors), '/', len(test_list))

for i in range(10):
    counter = 0
    for entry in filter(lambda x: x == i, errors):
        counter += 1
    print('Errors for ', i, ': ', counter, sep='')



