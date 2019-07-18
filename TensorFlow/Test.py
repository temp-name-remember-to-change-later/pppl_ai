import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100).astype(np.float32)
a = 50
b = 40
Y = a * X + b

Y = np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=10))(Y)
a_var = tf.Variable(1.0)
b_var = tf.Variable(1.0)
y_var = a_var * X + b_var

loss = tf.reduce_mean(tf.square(y_var - Y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

TRAINING_STEPS = 300
results = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(TRAINING_STEPS):
        results.append(sess.run([train, a_var, b_var])[1:])

final_pred = results[-1]
a_hat = final_pred[0]
b_hat = final_pred[1]
y_hat = a_hat * X + b_hat

print("a:", a_hat, "b:", b_hat)

plt.scatter(X, Y)
plt.plot(X, y_hat)
plt.show()
