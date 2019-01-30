from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.001
batch_size = 100
beta = 0.01

# Network Parameters
n_hidden_1 = 250 # number of hidden nodes in layer1
n_hidden_2 = 250
n_hidden_3 = 250
n_input = 784 # numer of input nodes
n_classes = 10 # number of output nodes

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


# weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#Set relation of layers
layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
layer_1 = tf.nn.relu(layer_1)
layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
layer_2 = tf.nn.relu(layer_2)
layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
layer_3 = tf.nn.relu(layer_3)
out_layer = tf.matmul(layer_3, weights['out']) + biases['out']

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=y))
#Regularization (comment the following two lines to cancel L2-regularization)
regularization = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['out']) +tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(weights['h3'])
loss_L2 = tf.reduce_mean(loss + beta * regularization)

#The first line does not implement L2-regularization, vise versa
#train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_L2)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    # Training cycle
    print ("Total : 20000")
    for i in range(20000):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        #don't foget to change the loss target here (loss / loss_L2)
        sess.run([train_step, loss_L2], feed_dict={x: batch_x, y: batch_y})
        if i % 2000 == 0:
            print ("Step: " + str(i))
    print ("Optimization Finished!")

    #Calculate Error
    correct_prediction = tf.equal(tf.argmax(out_layer, 1),  tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    te_error = 1-(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    tr_error = 1-(sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels}))
    print ("Testing Error: " + str(te_error))
    print ("Training Error: " + str(tr_error))

