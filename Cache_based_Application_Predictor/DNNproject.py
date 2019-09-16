#importing modules
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import copy



#********** Defining UserDefined Function ******************

# convert lebel to one hotCoder
def convert_to_onehot(array):
	row,col = np.shape(array)
	ls=[]
	for i in range(6):			# 15 defines the number of lebels
		ls.append(0)
	hot = []
	for i in range(row):
	 	s =copy.deepcopy(ls)
		#print [i,0]
		s[array[i,0]] = 1
		hot.append(s)
	return np.array(hot)


#********** STEP1 :  Importing Data ***********************

data=np.genfromtxt("MLDATA.csv", dtype="str",delimiter=",")
udata = data[1:,:]
np.random.shuffle(udata)

#print udata

x1_train = udata[:95,0].astype(float)
x2_train = udata[:95,1].astype(float)
'''
x3_train = udata[:95,5].astype(float)
x4_train = udata[:240,9].astype(float)
x5_train = udata[:240,6].astype(float)
x6_train = udata[:240,10].astype(float)
x7_train = udata[:240,7].astype(float)
x8_train = udata[:240,11].astype(float)
'''
x_train = np.c_[x1_train,x2_train]#,x5_train,x6_train,x3_train,x4_train]#,x7_train,x8_train]#]
y_train = udata[:95,2:3].astype(float)
y_train = y_train.astype(int)
x1_test =  udata[95:,0].astype(float)
x2_test =  udata[95:,1].astype(float)
'''
x3_test =  udata[240:,5].astype(float)
x4_test = udata[240:,9].astype(float)
x5_test = udata[240:,6].astype(float)
x6_test = udata[240:,10].astype(float)
x7_test = udata[240:,7].astype(float)
x8_test = udata[240:,11].astype(float)
'''
x_test = np.c_[x1_test,x2_test]#,x5_test,x6_test,x3_test,x4_test]#,x7_test,x8_test]#]
y_test = udata[95:,2:3].astype(float) 
y_test = y_test.astype(int) 

#********** STEP2 :  Selecting Model ***********************

# Hyperparameters Setting
learning_rate = 0.01
batch_size = 100
beta = 0.01
epoch =  500

# Network Parameters 
n_hidden_1 = 50              #number of hidden nodes in ith hidden layer   
n_hidden_2 = 50
n_hidden_3 = 50
n_input = 2 		      # numer of input nodes/features
n_classes = 6                # number of output nodes/classes

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

# Set relation of layers
layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
layer_1 = tf.nn.relu(layer_1)
layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
layer_2 = tf.nn.relu(layer_2)
layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
layer_3 = tf.nn.relu(layer_3)
out_layer = tf.matmul(layer_3, weights['out']) + biases['out']      # our model 


# Define loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=y))

##### Regularization loss
regularization = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['out']) +tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(weights['h3'])
loss_L2 = tf.reduce_mean(loss + beta * regularization)

# Define Optimizer
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_L2)  #select optimizer for regularization loss

# Initializing the variables
init = tf.global_variables_initializer()


#********** STEP3 :  Training Model ***********************

Loss_list1 = []
Loss_list2 = []
tracc_list = []
teacc_list = []
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    print ("Total : "+ str(epoch))
    for i in range(epoch):
	#print ("start", i)
	for batch in range(5):
       		batch_x  =  x_train[batch:batch+15]
		batch_y =   convert_to_onehot(y_train[batch:batch+15])
                #don't foget to change the loss target here (loss / loss_L2)
       		_,loss_value1= sess.run([train_step, loss], feed_dict={x: batch_x, y: batch_y})
		_,loss_value2= sess.run([train_step, loss], feed_dict={x: x_test, y: convert_to_onehot(y_test)})
		correct_prediction = tf.equal(tf.argmax(out_layer, 1),tf.argmax(y, 1))
        	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        	a1 = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
		a2 = sess.run(accuracy, feed_dict={x: x_test, y: convert_to_onehot(y_test)})
	Loss_list1.append(loss_value1)
	Loss_list2.append(loss_value2)
	tracc_list.append(a1)
	teacc_list.append(a2)
        if i % 5 == 0:
           print ("Step: " + str(i) + "| " + str(loss_value1) + " |" +str(a1))
	   print ("TStep: " + str(i) + "| " + str(loss_value2) + " |" +str(a2))
    print ("Optimization Finished!")



#********** STEP3 :  Visualization ***********************
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training & Testing Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(Loss_list1)#, "training loss", label ="training loss")
axes[0].plot(Loss_list2)#, "testing loss", label ="testing loss")

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(tracc_list)#, "training Accuracy", label ="training Accuracy")
axes[1].plot(teacc_list)#, "testing  Accuracy", label ="testing Accuracy")

'''
ax2 = axes[1].twiny()
ax2.plot(range(100),np.ones(100))
ax2.set_xlabel("timing", fontsize=14)
ax2.cla()
'''
plt.show()

 









