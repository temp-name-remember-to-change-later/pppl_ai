import tensorflow.compat.v1 as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()



# initialize parameters
learning_rate = 0.5
input_nodes = 28*28
hidden_nodes = 200
output_nodes = 10
init_stddev = 0.03

# initialize input and output placeholders
L1 = tf.placeholder('float', [input_nodes])
L2 = tf.placeholder('float', [hidden_nodes])
L3 = tf.placeholder('float', [output_nodes])

# initialize starting weights and biases with random normal distribution
W12 = tf.Variable(tf.random.normal([input_nodes, hidden_nodes], stddev=init_stddev))
W23 = tf.Variable(tf.random.normal([hidden_nodes, output_nodes], stddev=init_stddev))
b2 = tf.Variable(tf.random.normal([hidden_nodes], stddev=init_stddev))
b3 = tf.Variable(tf.random.normal([output_nodes], stddev=init_stddev))

# calculate hidden1 outputs from inputs; hidden2 outputs from hidden1; and final outputs from hidden2
# relu and softmax are 'squishification' functions
# matmul applies weights; add applies biases
hout = tf.nn.relu(tf.add(tf.matmul(L1, W12), b2))
oout = tf.nn.softmax(tf.add(tf.matmul(L2, W23), b3))

# clips values to prevent log(0) error and infinite log(1) chains
# uses cross entropy function to calculate cost via entropy
# optimizer back-propagates cost using earlier-defined learning rate and the cross_entropy function
oout = tf.clip_by_value(oout, 1e-10, 0.99999999)
costf = tf.losses.mean_squared_error(oout, )

# tf optimizer

# checks each prediction
# records accuracy

# runs session
