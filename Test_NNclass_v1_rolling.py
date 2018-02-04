
import os,sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''


import Network_class
import tensorflow as tf
import gzip, cPickle
import numpy as np

Train_batch_size = 64
Train_Glob_Iterations = 50
###################################################################################
def import_pickled_data(string):
    f = gzip.open('../data/'+string+'.pkl.gz','rb')
    dataset = cPickle.load(f)
    X_train = dataset[0]
    X_test  = dataset[1]
    y_train = dataset[2]
    y_test  = dataset[3]
    return X_train, y_train, X_test, y_test

sys.path.append('../HDR')
from library_HDR_v1 import *

def dimred_data(Sample1, Sample2, o_dim, g_size):
    Level, X_red_train = dim_reduction(Sample1, i_dim=Sample1.shape[1], o_dim =o_dim, \
        g_size=g_size, flag = 'corr')
    X_red_test=dim_reduction_test(Sample2, Level, i_dim=Sample2.shape[1], o_dim=o_dim,\
        g_size=g_size)
    return X_red_train, X_red_test

###################################################################################
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

###################################################################################
def return_dict(placeholder, List, model, batch_x, batch_y):

    S ={}
    for i, element in enumerate(List):
        S[placeholder[i]] = element
    S[model.Deep['FL_layer0']    ] = batch_x
    S[model.classifier['Target'] ] = batch_y
    return S

def sample_Z(X, m, n, kappa):
    return (X + 0*np.random.uniform(-kappa, kappa, size=[m, n]))
    # return (X +  np.random.normal(0, kappa, size=[m, n]))


#####################################################################################
def Analyse_custom_Optimizer(X_train, y_train, X_test, y_test,  kappa):
    import gc
    # Lets start with creating a model and then train batch wise.
    model = Network_class.Agent()
    model = model.init_NN_custom(classes, 0.01, [inputs,  50, 50], tf.nn.relu)
    a = np.zeros( ( (Train_Glob_Iterations/1) , 1))
    count = 0
    try:
        Noise_data = sample_Z(X_test, X_test.shape[0], X_test.shape[1], kappa = 0*kappa)
        t = xrange(Train_Glob_Iterations)
        from tqdm import tqdm
        for i in tqdm(t):
            for batch in iterate_minibatches(X_train, y_train, Train_batch_size, shuffle=True):
                batch_xs, batch_ys  = batch

                # Gather Gradients
                grads = model.sess.run([ model.Trainer["grads"] ],
                feed_dict ={ model.Deep['FL_layer0'] : batch_xs, model.classifier['Target']: batch_ys })
                List = [g for g in grads[0]]

                # Apply gradients
                summary, _ = model.sess.run( [ model.Summaries['merged'], model.Trainer["apply_placeholder_op"] ], \
                feed_dict= return_dict( model.Trainer["grad_placeholder"], List, model, batch_xs, batch_ys) )
                # model.Summaries['train_writer'].add_summary(summary, i)

            if i % 1 == 0:
                Nada, a[count]  = model.sess.run( [model.Summaries['merged'], model.Evaluation['accuracy']], feed_dict={ model.Deep['FL_layer0'] : \
                Noise_data, model.classifier['Target'] : y_test})
                print("Accuracy", a[count])

                # model.Summaries['test_writer'].add_summary(summary, i)
                count = count+1;
            if max(a) > 0.99:
                summary, pr  = model.sess.run( [ model.Summaries['merged'], model.Evaluation['prob'] ], \
                feed_dict ={ model.Deep['FL_layer0'] : X_test, model.classifier['Target'] : y_test } )
                break
                # model.Summaries['test_writer'].add_summary(summary, i)

    except Exception as e:
        print e
        print "I found an exception"
        tf.reset_default_graph()
        del model
        gc.collect()
        return 0

    tf.reset_default_graph()
    del model
    gc.collect()
    print("Accuracy is", a[count-1])
    return a[count-1]


# Mnist Dataset
# from tensorflow.examples.tutorials.mnist import input_data
# Temp =[]
# from tqdm import tqdm
# filename = 'MNIST_single_kappa_gauss.csv'
# print("The test running is", filename)
# # X_train, y_train, X_test, y_test = import_pickled_data(dataset)
# mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
# X_train = mnist.train.images
# y_train = mnist.train.labels
# X_test  = mnist. test.images
# y_test  = mnist.test.labels
# from sklearn import preprocessing
# X_train = preprocessing.scale(X_train)
# X_test = preprocessing.scale(X_test)
# inputs = X_train.shape[1]
# classes = y_train.shape[1]
##  Sensorless

Temp = []
import tflearn
from tqdm import tqdm
X_train, y_train, X_test, y_test = import_pickled_data('gisette')
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# X_train, X_test = dimred_data(X_train, X_test, 500, 100)
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
x = input()

classes = int(max(y_train)+1)
print("classes", classes)
y_train = tflearn.data_utils.to_categorical((y_train), classes)
y_test  = tflearn.data_utils.to_categorical((y_test), classes)
from sklearn import preprocessing
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)
print "Train, Test, arcene", X_train.shape, X_test.shape
inputs  = X_train.shape[1]
classes = y_train.shape[1]
filename = 'Cifar10_Nonoise.csv'

## Final Executions
print(filename)
iterat_kappa = 1
Kappa_s = np.random.uniform(0, 1, size=[iterat_kappa])
for i in tqdm(xrange(iterat_kappa)):
    Temp.append(Analyse_custom_Optimizer(X_train,y_train,X_test,y_test, Kappa_s[i]))
Results = np.zeros([iterat_kappa,2])
Results[:,1] = Temp
print(Results[:,1].mean(), Results[:,1].std())
Results[:,0] = Kappa_s[:]
# np.savetxt(filename, Results, delimiter=',')
