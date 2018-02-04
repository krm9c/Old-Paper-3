import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def sample_Z(X, m, n, kappa):
    # return (X + np.random.uniform(-kappa, kappa, size=[m, n]))
    return (X + np.random.normal(0, kappa, size=[m, n]))

def single_NN():
    # NN -1 
    X = tf.placeholder(tf.float32, shape=[None, 784])
    D_W1 = tf.Variable(xavier_init([784, 128]))
    D_b1 = tf.Variable(tf.zeros(shape=[128]))
    D_W2 = tf.Variable(xavier_init([128, 10]))
    D_b2 = tf.Variable(tf.zeros(shape=[10]))
    theta_D = [D_W1, D_W2, D_b1, D_b2]
    Targets = tf.placeholder(tf.float32, shape=[None, 10])
    def discriminator(x):
        D_h1 = tf.tanh(tf.matmul(x, D_W1) + D_b1)
        D_logit = tf.matmul(D_h1, D_W2) + D_b2
        return D_logit
    D_logit_NN_1 = discriminator(X)
    D_loss_NN1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= ( D_logit_NN_1) , labels=Targets))
    D_NN_1_solver = tf.train.AdamOptimizer().minimize(D_loss_NN1, var_list=theta_D)

    # Lets make some predictions
    correct_Prediction = tf.equal(tf.argmax( (tf.nn.softmax(D_logit_NN_1)),1),\
                        tf.argmax(Targets,1))
    accuracy = tf.reduce_mean(tf.cast(correct_Prediction, tf.float32))

    mb_size = 128
    Z_dim = 784
    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if not os.path.exists('out/'):
        os.makedirs('out/')
    i = 0
    X_mb    = mnist.train.images
    YTr_Lab = mnist.train.labels
    XTr     = sample_Z( X_mb, X_mb.shape[0] , X_mb.shape[1], 0.3)
    for it in range(10000):
        X_mb, Y_mb = mnist.train.next_batch(mb_size)
        _, D_loss_curr = sess.run([D_NN_1_solver, D_loss_NN1], feed_dict={X: X_mb, Targets: Y_mb})
        if it % 1000 == 0:
            acc_tr = sess.run([accuracy], feed_dict ={X:XTr, Targets:YTr_Lab })
            acc_te = sess.run([accuracy], feed_dict ={X:mnist.test.images, Targets:mnist.test.labels } )
            print('Iter: {}'.format(it), 'NN loss: {:.4}'. format(D_loss_curr), 'Accuracy:', 'Train', acc_tr, 'Test', acc_te)


def double_NN():
    # NN -1 
    X = tf.placeholder(tf.float32, shape=[None, 784])
    D_W1 = tf.Variable(xavier_init([784, 128]))
    D_b1 = tf.Variable(tf.zeros(shape=[128]))
    D_W2 = tf.Variable(xavier_init([128, 10]))
    D_b2 = tf.Variable(tf.zeros(shape=[10]))
    theta_D = [D_W1, D_W2, D_b1, D_b2]

    # NN-2
    Z = tf.placeholder(tf.float32, shape=[None, 784])
    G_W1 = tf.Variable(xavier_init([784, 128]))
    G_b1 = tf.Variable(tf.zeros(shape=[128]))
    G_W2 = tf.Variable(xavier_init([128, 10]))
    G_b2 = tf.Variable(tf.zeros(shape=[ 10 ]))
    theta_G = [G_W1, G_W2, G_b1, G_b2]
    Targets = tf.placeholder(tf.float32, shape=[None, 10])
    def generator(z):
        G_h1 = tf.tanh(tf.matmul(z, G_W1) + G_b1)
        G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
        return G_log_prob
    def discriminator(x):
        D_h1 = tf.tanh(tf.matmul(x, D_W1) + D_b1)
        D_logit = tf.matmul(D_h1, D_W2) + D_b2
        return D_logit

    # Define the network outputs
    D_logit_NN_1 = discriminator(X)
    D_logit_NN_2 = generator(Z)
    
    # Define the losses
    D_loss_NN1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= ( D_logit_NN_1) , labels=Targets))
    D_loss_NN2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= ( D_logit_NN_2) , labels=Targets) )
    D_NN_1_solver = tf.train.AdamOptimizer().minimize(  (D_loss_NN1)-0.1*D_loss_NN1*D_loss_NN2, var_list=(theta_D) )
    D_NN_2_solver = tf.train.AdamOptimizer().minimize(  (D_loss_NN2)-0.1*D_loss_NN1*D_loss_NN2, var_list=(theta_G) )
    
    # Lets make some predictions
    correct_Prediction = tf.equal(tf.argmax( 0.5*(tf.nn.softmax(D_logit_NN_1)+tf.nn.softmax(D_logit_NN_2)),1),\
                         tf.argmax(Targets,1))
    accuracy = tf.reduce_mean(tf.cast(correct_Prediction, tf.float32)) 
    mb_size = 128
    Z_dim   = 784
    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    if not os.path.exists('out/'):
        os.makedirs('out/')
    i = 0
    X_mb = mnist.train.images
    YTr_Lab = mnist.train.labels
    XTr = sample_Z( X_mb, X_mb.shape[0] , X_mb.shape[1], 0.3)

    for it in range(20000):
        X_mb, Y_mb = mnist.train.next_batch(mb_size)
        Z_mb = sample_Z(X_mb, mb_size, Z_dim,1)
        _ = sess.run([D_NN_1_solver], feed_dict={X: X_mb, Z: Z_mb,Targets: Y_mb})
        _ = sess.run([D_NN_2_solver], feed_dict={X: X_mb, Z: Z_mb,Targets: Y_mb})
        if it % 100 == 0:
            acc_tr = sess.run([accuracy], feed_dict ={X:XTr, Z:XTr, Targets:YTr_Lab })
            acc_te = sess.run([accuracy], feed_dict ={X:mnist.train.images,  Z:mnist.train.images , Targets:mnist.train.labels } )
            print('Iter: {}'.format(it), 'Accuracy:', 'Train', acc_tr, 'Test', acc_te)

# single_NN()
double_NN()