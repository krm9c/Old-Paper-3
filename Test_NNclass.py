import Network_class
import tensorflow as tf
import gzip, cPickle
import numpy as np



###################################################################################
def import_pickled_data(string):
    f = gzip.open('../data/'+string+'.pkl.gz','rb')
    dataset = cPickle.load(f)
    X_train = dataset[0]
    X_test  = dataset[1]
    y_train = dataset[2]
    y_test  = dataset[3]
    return X_train, y_train, X_test, y_test

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


#####################################################################################
def Analyse_custom_Optimizer(X_train, y_train, X_test, y_test):

    import gc
    # Lets start with creating a model and then train batch wise.
    model = Network_class.Agent()
    model = model.init_NN_custom(classes, 0.001, [inputs, 300, 300, 300, 300, 300, 300], tf.nn.relu)

    try:
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
                summary, a  = model.sess.run( [model.Summaries['merged'], model.Evaluation['accuracy']], feed_dict={ model.Deep['FL_layer0'] : \
                X_test, model.classifier['Target'] : y_test})
                # print "accuracy -- ", a
                # model.Summaries['test_writer'].add_summary(summary, i)
            if a > 0.99:
                summary, pr  = model.sess.run( [ model.Summaries['merged'], model.Evaluation['prob'] ], \
                feed_dict ={ model.Deep['FL_layer0'] : X_test, model.classifier['Target'] : y_test } )
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
    return a


Train_batch_size = 256
Train_Glob_Iterations = 200
dataset = 'rolling'
Temp =[]

from tqdm import tqdm
X_train, y_train, X_test, y_test = import_pickled_data(dataset)
from sklearn import preprocessing
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

print "Train, Test", X_train.shape, X_test.shape


x = input();
import os,sys
sys.path.append('../CommonLibrariesDissertation')
from Library_Paper_two import *


X_train, Tree = initialize_calculation(T = None, Data = X_train, gsize = 2,\
par_train = 0, output_dimension = 200)
X_test, Tree = initialize_calculation(T = Tree, Data = X_test, gsize = 2,\
par_train = 1, output_dimension = 200)
print "Train, Test", X_train.shape, X_test.shape
inputs = X_train.shape[1]
classes = (max(y_train))
print classes
import tflearn
for i in tqdm(xrange(100)):
    Temp.append(Analyse_custom_Optimizer(X_train,\
     tflearn.data_utils.to_categorical((y_train-1), classes),\
      X_test, tflearn.data_utils.to_categorical((y_test-1), classes)))
Results = np.array(Temp)
print "\n min", min(Results), "avg", Results.mean(), "max", max(Results)
