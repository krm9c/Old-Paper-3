import Test_paper_3_NN as  NN_class
import tensorflow as tf
import gzip, cPickle
import numpy as np
import traceback
from tensorflow.examples.tutorials.mnist import input_data

Train_batch_size = 64
Train_Glob_Iterations = 30
dataset = 'rolling'
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
def return_dict(placeholder, List, model, batch_x, batch_x_noise, batch_y):

    S ={}
    for i, element in enumerate(List):
        S[placeholder[i]] = element
    S[model.Deep_1['FL_layer0']    ] = batch_x
    S[model.Deep_2['FL_layer0']    ] = batch_x_noise
    S[model.classifier['Target'] ] = batch_y
    return S

def sample_Z(X, m, n, kappa):
    # return (X + np.random.uniform(-kappa, kappa, size=[m, n]))
    return (X +  np.random.normal(0, kappa, size=[m, n]))

#####################################################################################
def Analyse_custom_Optimizer(X_train, y_train, X_test, y_test, kappa):
    import gc
    # Lets start with creating a model and then train batch wise.
    model = NN_class.learners()
    model = model.init_NN_custom(classes, 0.01, [inputs, 50, 50, 50,50,50 ], tf.nn.relu)
    a = np.zeros( ( (Train_Glob_Iterations/1) , 1))
    try:
        count = 0        
        t = xrange(Train_Glob_Iterations)
        from tqdm import tqdm
        Noise_data = sample_Z(X_test, X_test.shape[0], X_test.shape[1], kappa = kappa)
        for i in tqdm(t):
            for batch in iterate_minibatches(X_train, y_train, Train_batch_size, shuffle=True):
                batch_xs, batch_ys  = batch   
                batch_noise_xs  = sample_Z(batch_xs, Train_batch_size, X_train.shape[1], 1)

                # Gather Gradients
                grads = model.sess.run([ model.Trainer["grads"] ],
                feed_dict ={ model.Deep_1['FL_layer0'] : batch_xs, model.Deep_2['FL_layer0'] : \
                batch_noise_xs, model.classifier['Target']: batch_ys })
                List = [g for g in grads[0]]
                # Apply gradients
                summary, _ = model.sess.run( [ model.Summaries['merged'], model.Trainer["apply_placeholder_op"] ], \
                feed_dict= return_dict( model.Trainer["grad_placeholder"], List, model, batch_xs, batch_noise_xs, batch_ys) )     
                #model.Summaries['train_writer'].add_summary(summary, i)

            if i % 1 == 0:
                summary, a[count]  = model.sess.run( [model.Summaries['merged'], model.Evaluation['accuracy']], feed_dict={ model.Deep_1['FL_layer0'] : \
                Noise_data, model.Deep_2['FL_layer0']: Noise_data, model.classifier['Target'] : y_test})
                # summary, a_train  = model.sess.run( [model.Summaries['merged'], model.Evaluation['accuracy']], feed_dict={ model.Deep_1['FL_layer0'] : \
                # X_train, model.Deep_2['FL_layer0']: X_train, model.classifier['Target'] : y_train})
                # print("The accuracy is",a, a_train)
                # model.Summaries['test_writer'].add_summary(summary, i)
                count = count+1;
                if max(a) > 0.99:
                    summary, pr  = model.sess.run( [ model.Summaries['merged'], model.Evaluation['prob'] ], \
                    feed_dict ={ model.Deep_1['FL_layer0'] : X_test,\
                    model.Deep_2['FL_layer0']:X_test,model.classifier['Target'] : y_test } )
                    break
                # model.Summaries['test_writer'].add_summary(summary, i)

    except Exception as e:
        print e
        print "I found an exception"
        traceback.print_exc()
        tf.reset_default_graph()
        del model
        gc.collect()
        return 0
    tf.reset_default_graph()
    del model
    gc.collect()
    return a[(count-1)]

Temp =[]
from tqdm import tqdm
# X_train, y_train, X_test, y_test = import_pickled_data(dataset)
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
X_train = mnist.train.images
y_train = mnist.train.labels
X_test  = mnist. test.images
y_test  = mnist.test.labels
from sklearn import preprocessing
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)
print "Train, Test, Mnist", X_train.shape, X_test.shape
inputs  = X_train.shape[1]
classes = y_train.shape[1]
filename = 'MNIST_Double_kappa_gauss.csv'
# print("classes", classes)
# x = input()
# import tflearn
# for i in tqdm(xrange(1)):
#     Temp.append(Analyse_custom_Optimizer(X_train,\
#      tflearn.data_utils.to_categorical((y_train), classes),\
#       X_test, tflearn.data_utils.to_categorical((y_test), classes)))
# Results = np.array(Temp)
# print "\n min", min(Results), "avg", Results.mean(), "max", max(Results)

print("classes", classes)
print("filename", filename)
iterat_kappa = 100
Kappa_s = np.random.uniform(0, 1, size=[iterat_kappa])
for i in tqdm(xrange(iterat_kappa)):
    Temp.append(Analyse_custom_Optimizer(X_train,y_train,X_test,y_test, Kappa_s[i]))
Results = np.zeros([iterat_kappa,2])
Results[:,1] = Temp
Results[:,0] = Kappa_s[:]
np.savetxt(filename, Results, delimiter=',')

