import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# initialize parameters
learning_rate = 0.5
hl_nodes = 50
epochs = 10
batch_size = 100
init_stddev = 0.03

# initialize input and output placeholders
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# initialize starting weights and biases with random normal distribution
W1 = tf.Variable(tf.random_normal([784, hl_nodes], stddev=init_stddev, name='W1'))
W2 = tf.Variable(tf.random_normal([hl_nodes, hl_nodes], stddev=init_stddev, name='W2'))
W3 = tf.Variable(tf.random_normal([hl_nodes, 10], stddev=init_stddev, name='W3'))

b1 = tf.Variable(tf.random_normal([hl_nodes]), name='b1')
b2 = tf.Variable(tf.random_normal([hl_nodes]), name='b2')
b3 = tf.Variable(tf.random_normal([10]), name='b3')


# calculate hidden1 outputs from inputs; hidden2 outputs from hidden1; and final outputs from hidden2
# relu and softmax are 'squishification' functions
# matmul applies weights; add applies biases
hidden1_out = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
hidden2_out = tf.nn.relu(tf.add(tf.matmul(hidden1_out, W2), b2))
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden2_out, W3), b3))

# clips values to prevent log(0) error and infinite log(1) chains
# uses cross entropy function to calculate cost via entropy
# optimizer back-propagates cost using earlier-defined learning rate and the cross_entropy function
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# tf optimizer
init_op = tf.global_variables_initializer()

# checks each prediction
# records accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# runs session
with tf.Session() as sess:
    sess.run(init_op)
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

