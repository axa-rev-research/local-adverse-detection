
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six.moves import xrange

import logging
import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils import to_categorical
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation

from cleverhans_tutorials.tutorial_models import make_basic_cnn, MLP
from cleverhans_tutorials.tutorial_models import Flatten, Linear, ReLU, Softmax
from cleverhans.utils import TemporaryLogLevel

from lad import lad_Thibault as lad
from scipy.spatial.distance import euclidean


import copy
import pandas
import numpy
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import euclidean_distances

"""
This tutorial shows how to generate adversarial examples
using FGSM in black-box setting.
The original paper can be found at:
https://arxiv.org/abs/1602.02697
"""

FLAGS = flags.FLAGS


# # Functions

# ## Data

# In[10]:


'''
MOONS
'''
def get_moon():
    X, y = make_moons(noise=0.3, random_state=1, n_samples=10000)
    y2 = numpy.zeros((X.shape[0],2))
    for k in range(len(y)):
        y2[k][y[k]] = 1
    return X, y2

def get_german():
    path_dataset='data/germancredit.csv'
    X = pandas.read_csv(path_dataset, delimiter=",", index_col=0)
    y = X.label
    y = y - 1
    X = X.iloc[:,X.columns != 'label']
    X = (X-X.mean())/X.std()
    y2 = numpy.zeros((X.shape[0],2)) #2=  nb de classes
    for k in range(len(y)):
        y2[k][y[k]] = 1
    return numpy.array(X), numpy.array(y2)

DATASETS_ = {'moons':get_moon,
            'german': get_german}


# ## Training a black-box

# In[3]:


'''
PAPERNOT BB
'''
def Papernot_bbox(sess, x, y, X_train, Y_train, X_test, Y_test,
              nb_epochs, batch_size, learning_rate,
              rng):
    """
    Define and train a model that simulates the "remote"
    black-box oracle described in the original paper.
    :param sess: the TF session
    :param x: the input placeholder for MNIST
    :param y: the ouput placeholder for MNIST
    :param X_train: the training data for the oracle
    :param Y_train: the training labels for the oracle
    :param X_test: the testing data for the oracle
    :param Y_test: the testing labels for the oracle
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param rng: numpy.random.RandomState
    :return:
    """

    # Define TF model graph (for the black-box model)
    model = make_basic_cnn()
    predictions = model(x)
    print("Defined TensorFlow model graph.")

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    model_train(sess, x, y, predictions, X_train, Y_train,
                args=train_params, rng=rng)

    # Print out the accuracy on legitimate data
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                          args=eval_params)
    print('Test accuracy of black-box on legitimate test '
          'examples: ' + str(accuracy))

    return model, predictions, accuracy

def RF_bbox(X_train, Y_train, X_test, Y_test):
    # Define RF model graph (for the black-box model)

    model = RandomForestClassifier(n_estimators=100, n_jobs=-1).fit(X_train, Y_train)
    
    # Print out the accuracy on legitimate data
    #predictions = model.predict_proba(X_test)[1] TEST CHANGER PREDICTIONS > FONCTION
    predictions=lambda x: model.predict_proba(x)[1] #predict_proba required ou alors changer du code (argmax et compagnie) de papernot
    
    accuracy = accuracy_score(Y_test, model.predict(X_test))
    #roc_auc = roc_auc_score(Y_test, predictions[1][:,1])
    print('Test accuracy of black-box on legitimate test '
          'examples: ' + str(accuracy))
    #print('Test ROC AUC of black-box on legitimate test ' 'examples: ' + str(roc_auc))
        
    
    return model, predictions, accuracy
    
BB_MODELS_ = {'dnn': Papernot_bbox,
            'rf': RF_bbox}
#ne pas utiliser dnn ca marche pas pour le moment


# ## Papernot Surrogate

# In[12]:


def setup_tutorial():
    """
    Helper function to check correct configuration of tf for tutorial
    :return: True if setup checks completed
    """

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    return True
def substitute_model(img_rows=1, img_cols=2, nb_classes=2):
    """
    Defines the model architecture to be used by the substitute. Use
    the example model interface.
    :param img_rows: number of rows in input
    :param img_cols: number of columns in input
    :param nb_classes: number of classes in output
    :return: tensorflow model
    """
    input_shape = (None, img_rows, img_cols, 1) #on garde format d'origine parce qu'on comprend pas grand chose mais on change valeurs

    # Define a fully connected model (it's different than the black-box)
    '''layers = [Flatten(),
              Linear(200),
              ReLU(),
              Linear(200),
              ReLU(),
              Linear(nb_classes),
              Softmax()]'''
    layers = [Flatten(), Linear(nb_classes), Softmax()] #surrogate simplifié

    return MLP(layers, input_shape)


def train_sub(sess, x, y, bb_model, X_sub, Y_sub, nb_classes,
              nb_epochs_s, batch_size, learning_rate, data_aug, lmbda,
              rng):
    """
    This function creates the substitute by alternatively
    augmenting the training data and training the substitute.
    :param sess: TF session
    :param x: input TF placeholder
    :param y: output TF placeholder
    :param bbox_preds: output of black-box model predictions
    :param X_sub: initial substitute training data
    :param Y_sub: initial substitute training labels
    :param nb_classes: number of output classes
    :param nb_epochs_s: number of epochs to train substitute model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param data_aug: number of times substitute training data is augmented
    :param lmbda: lambda from arxiv.org/abs/1602.02697
    :param rng: numpy.random.RandomState instance
    :return:
    """
    # Define TF model graph (for the black-box model)
    model_sub = substitute_model(img_cols=X_sub.shape[1])
    preds_sub = model_sub(x)
    print("Defined TensorFlow model graph for the substitute.")

    # Define the Jacobian symbolically using TensorFlow
    grads = jacobian_graph(preds_sub, x, nb_classes)
    # Train the substitute and augment dataset alternatively
    for rho in xrange(data_aug):
        print("Substitute training epoch #" + str(rho))
        train_params = {
            'nb_epochs': nb_epochs_s,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        with TemporaryLogLevel(logging.WARNING, "cleverhans.utils.tf"):
            model_train(sess, x, y, preds_sub, X_sub,
                        to_categorical(Y_sub, nb_classes),
                        init_all=False, args=train_params, rng=rng)

        # If we are not at last substitute training iteration, augment dataset
        if rho < data_aug - 1:
            print("Augmenting substitute training data.")
            # Perform the Jacobian augmentation
            lmbda_coef = 2 * int(int(rho / 3) != 0) - 1
            X_sub = jacobian_augmentation(sess, x, X_sub, Y_sub, grads,
                                          lmbda_coef * lmbda)
            
            print("Labeling substitute training data.")
            # Label the newly generated synthetic points using the black-box
            Y_sub = numpy.hstack([Y_sub, Y_sub])
            X_sub_prev = X_sub[int(len(X_sub)/2):] #on a double le dataset donc prev = ce qu'il y a de nouveau = la moitie
            eval_params = {'batch_size': batch_size}
            
            #bbox_preds = tf.convert_to_tensor(bbox_preds, dtype=tf.float32) TEST CHANGER PREDICTIONS > FONCTION           
            #bbox_val = batch_eval2(sess, [x], [bbox_preds], [X_sub_prev], args=eval_params)[0] TEST CHANGER PREDICTIONS > FONCTION
            
            #bbox_val = bbox_preds(X_sub_prev) #normalement batch eval sert juste à sortir les preds...?
            bbox_val = bb_model.predict(X_sub_prev)
            # Note here that we take the argmax because the adversary
            # only has access to the label (not the probabilities) output
            # by the black-box model
            Y_sub[int(len(X_sub)/2):] = numpy.argmax(bbox_val, axis=1)
    return model_sub, preds_sub 


# Usage: 
# print("Training the substitute model.")
#     train_sub_out = train_sub(sess, x, y, bbox_preds, X_sub, Y_sub,
#                               nb_classes, nb_epochs_s, batch_size,
#                               learning_rate, data_aug, lmbda, rng=rng)
#     model_sub, preds_sub = train_sub_out

# # Our surrogate

# # Local Fidelity

# In[6]:


def get_random_points_hypersphere(x_center, radius_, n_points_):

        res = []
        while len(res) < n_points_:
        
            n_points_left_ = n_points_ - len(res)
            # About half the points are lost in the test hypercube => hypersphere
            lbound = numpy.repeat([x_center.values-(radius_/2.)], n_points_left_*2, axis=0)
            hbound = numpy.repeat([x_center.values+(radius_/2.)], n_points_left_*2, axis=0)
            points = numpy.random.uniform(low=lbound, high=hbound)
            # Check if x_generated is within hypersphere (if kind=='hypersphere')
            for x_generated in points:
                if euclidean(x_generated, x_center.values) < radius_:
                    res.append(x_generated)
                if len(res) == n_points_:
                    break

        return pandas.DataFrame(numpy.array(res))
    
def generate_inside_ball(center, segment=(0,1), n=1): #verifier algo comprendre racine 1/d et rapport entre segment et radius
    d = center.shape[0]
    z = numpy.random.normal(0, 1, (n, d))
    z = numpy.array([a * b / c for a, b, c in zip(z, numpy.random.uniform(*segment, n),  norm(z))])
    z = z + center
    return z 
def norm(v):
        return numpy.linalg.norm(v, ord=2, axis=1) #array of l2 norms of vectors in v


# # Framework

# In[14]:


nb_classes=2 #
batch_size=20 #
learning_rate=0.001 #
nb_epochs=0 # Nombre d'itération bbox osef
holdout=50 # Nombre d'exemples utilisés au début pour générer data (Pap-substitute)
data_aug=6 # Nombre d'itérations d'augmentation du dataset {IMPORTANT pour Pap-substitute}
nb_epochs_s=10 # Nombre d'itérations pour train substitute
lmbda=0.1 # params exploration pour augmentation data
#radius_ = 0.5 # NEW


#main_fidelity(radius_)


# 
# Il faut trouver une facon de faire la boucle
# 
# pour radius:
#     genere black box
#     genere surrogate papernot
#     
#     pour observation dans test:
#         genere local surrogate
#         evalue papernot local
#         evalue local surrogate local
# outputs:
# papernot: {radius: [accuracy locale de chaque point}
# pareil pour ls}
# 
# 
# TODO: check histoire de boucle radius comment ca se goupille
# voir si ca tourne
# faire graphe...
# 

# In[15]:


# Seed random number generator so tutorial is reproducible
rng = numpy.random.RandomState([2017, 8, 30])

# Thibault: Tensorflow stuff
set_log_level(logging.DEBUG)
assert setup_tutorial()
sess = tf.Session()



# Data
X, Y = DATASETS_['german']()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)
X_sub = X_test[:holdout]
Y_sub = numpy.argmax(Y_test[:holdout], axis=1)

## Redefine test set as remaining samples unavailable to adversaries
### N.B Thibault: c'est pour le substitute de Papernot
X_test = X_test[holdout:]
Y_test = Y_test[holdout:]
print("Training black box on",X_train.shape[0], "examples")
print('Testing black box and substitute on', X_test.shape[0],' examples')
print("Using ", holdout, " examples to start PP substitute")
## Define input and output TF placeholders
### N.B. Thibault: restes de Tensorflow, utilisé pour le substitute de Papernot...
x = tf.placeholder(tf.float32, shape=(None, X.shape[1]))
y = tf.placeholder(tf.float32, shape=(None, Y.shape[1]))  

# Simulate the black-box model
print("Preparing the black-box model.")
prep_bbox_out = BB_MODELS_['rf'](X_train, Y_train, X_test, Y_test)
bb_model, bbox_preds, _ = prep_bbox_out #bbox_preds fonction predict

# Train PAPERNOT substitute
print("Training the Pépèrenot substitute model.")
train_sub_pap = train_sub(sess, x, y, bb_model, X_sub, Y_sub,
                          nb_classes, nb_epochs_s, batch_size,
                          learning_rate, data_aug, lmbda, rng=rng)
model_sub, preds_sub = train_sub_pap

eval_params = {'batch_size': batch_size}
pap_acc = model_eval(sess, x, y, preds_sub, X_test, Y_test, args=eval_params) 
print(pap_acc)







# In[16]:


import copy
from multiprocessing import Pool



def pred(x):
        return bb_model.predict(x)[:,1]

xs_toexplain = [pandas.Series(xi) for xi in X_test]
radius_perc=[0.05,0.1,0.2,0.3,0.4,0.5]#,0.6,0.7,0.8,0.9,1] 
papernot = {}
localsurr = {}
papernot = dict([(r, []) for r in radius_perc])
localsurrogate = dict([(r, []) for r in radius_perc])
c = 0



for x_toexplain in xs_toexplain:
    c += 1
    if c % 10 == 0:
        print('iter', c)
    
    print("Training Local Surrogate substitute model.")
    
    
    _, train_sub_ls = lad.LocalSurrogate(pandas.DataFrame(X), blackbox=bb_model, n_support_points=1000, max_depth=5).get_local_surrogate(x_toexplain)
    
    print("Calculating distances.")
    dists = euclidean_distances(x_toexplain.to_frame().T, X)
    #dists = pandas.Series(dists[0], index=X.index)
    radius_all_ = dists.max()*numpy.array(radius_perc)

    
    for i in range(len(radius_all_)):
        radius = radius_all_[i]
        #support_x_ = numpy.array(get_random_points_hypersphere(x_toexplain, radius_=radius, n_points_=1000))
        support_x_ = generate_inside_ball(numpy.array(x_toexplain), segment=(0, radius), n=1000)
        

        pap_fid = model_eval(sess, x, y, preds_sub, support_x_, bb_model.predict(support_x_) , args=eval_params)
        papernot[radius_perc[i]].append(pap_fid)

        ls_fid = accuracy_score(train_sub_ls.predict(support_x_), pred(support_x_))
        localsurrogate[radius_perc[i]].append(ls_fid)





# In[20]:


out_localsurr = pandas.DataFrame(localsurrogate)
out_papernot = pandas.DataFrame(papernot)
out_localsurr.to_csv('results/tables/exp1_german_localsurr5.csv')
out_localsurr.to_csv('results/tables/exp1_german_papernot5.csv')


import seaborn as sns
import matplotlib.pyplot as plt

sns.pointplot(data=out_papernot)
sns.pointplot(data=out_localsurr, color='orange')
plt.xlabel('Radius percent')
plt.ylabel('Local Accuracy')
plt.savefig('results/figures/local_fidelity_german5.pdf')

