
# coding: utf-8
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

def get_spambase():
    path_dataset='data/spambase.csv'
    X = pandas.read_csv(path_dataset, delimiter=",", index_col=None)
    y = X.label
    X = X.iloc[:,X.columns != 'label']
    X = (X-X.mean())/X.std()
    y2 = numpy.zeros((X.shape[0],2)) #2=  nb de classes
    for k in range(len(y)):
        y2[k][y[k]] = 1
    return numpy.array(X), numpy.array(y2)
    

DATASETS_ = {'moons':get_moon,
            'german': get_german,
            'spambase': get_spambase}


def pred(x):
        return bb_model.predict(x)[:,1]



'''
Black-box
'''
def RF_bbox(X_train, Y_train, X_test, Y_test):
    # Define RF model (for the black-box model)
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1).fit(X_train, Y_train)
    
    predictions=lambda x: model.predict_proba(x)[1] #predict_proba required ou alors changer du code (argmax et compagnie) de papernot
        accuracy = accuracy_score(Y_test, model.predict(X_test))
    print('Test accuracy of black-box on legitimate test '
          'examples: ' + str(accuracy))
    return model, predictions, accuracy
    
BB_MODELS_ = {'rf': RF_bbox}


'''
Trucs que je comprends moins/peu/pas fournis par Papernot et modifiés "juste pour que ça colle"
'''
def setup_tutorial():
    """
    Helper function to check correct configuration of tf for tutorial
    :return: True if setup checks completed
    """
    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)
    return True

def PAP_substitute_model(img_rows=1, img_cols=2, nb_classes=2):
    """
    Defines the model architecture to be used by the substitute. Use
    the example model interface.
    :param img_rows: number of rows in input
    :param img_cols: number of columns in input
    :param nb_classes: number of classes in output
    :return: tensorflow model
    """
    input_shape = (None, img_rows, img_cols, 1) #code sous cette forme: vient de Papernot, je garde pour pas tout casser...
    # Define a fully connected model (it's different than the black-box)
    layers = [Flatten(),
              Linear(200),
              ReLU(),
              Linear(200),
              ReLU(),
              Linear(nb_classes),
              Softmax()]
    #layers = [Flatten(), Linear(nb_classes), Softmax()] #surrogate simplifié
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
            bbox_val = bb_model.predict(X_sub_prev)
            Y_sub[int(len(X_sub)/2):] = numpy.argmax(bbox_val, axis=1)
    return model_sub, preds_sub 


'''
Nouvelle fonction générer dans boule
'''
def generate_inside_ball(center, segment=(0,1), n=1):
    def norm(v):
        return numpy.linalg.norm(v, ord=2, axis=1) #array of l2 norms of vectors in v
    d = center.shape[0]
    z = numpy.random.normal(0, 1, (n, d))
    z = numpy.array([a * b / c for a, b, c in zip(z, numpy.random.uniform(*segment, n),  norm(z))])
    z = z + center
    return z 




'''
Paramètres
'''
nb_classes=2 #
batch_size=20 #
learning_rate=0.001 #
holdout=50 # Nombre d'exemples utilisés au début pour générer data (Pap-substitute)
data_aug=6 # Nombre d'itérations d'augmentation du dataset {IMPORTANT pour Pap-substitute}
nb_epochs_s=10 # Nombre d'itérations pour train substitute
lmbda=0.1 # params exploration pour augmentation data


# Seed random number generator so tutorial is reproducible
rng = numpy.random.RandomState([2017, 8, 30])



'''
Re des trucs tf peu clairs
'''set_log_level(logging.DEBUG)
assert setup_tutorial()
sess = tf.Session()



'''
Framework: black box + papernot substitute
'''
dname = 'german'
X, Y = DATASETS_[dname]()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
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
bb_model, bbox_preds, _ = prep_bbox_out

# Train PAPERNOT substitute
print("Training the Pépèrenot substitute model.")
train_sub_pap = train_sub(sess, x, y, bb_model, X_sub, Y_sub,
                          nb_classes, nb_epochs_s, batch_size,
                          learning_rate, data_aug, lmbda, rng=rng)
model_sub, preds_sub = train_sub_pap

eval_params = {'batch_size': batch_size}
pap_acc = model_eval(sess, x, y, preds_sub, X_test, Y_test, args=eval_params) 
print(pap_acc)



xs_toexplain = [pandas.Series(xi) for xi in X_test]
radius_perc=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] 
papernot = {}
localsurr = {}
papernot = dict([(r, []) for r in radius_perc])
localsurrogate = dict([(r, []) for r in radius_perc])
c = 0



for x_toexplain in xs_toexplain:
    c += 1 #counter
    if c % 10 == 0:
        print('iter', c)
    
    print("Training Local Surrogate substitute model.")
        
    _, train_sub_ls = lad.LocalSurrogate(pandas.DataFrame(X), blackbox=bb_model, n_support_points=1000, max_depth=5).get_local_surrogate(x_toexplain)
    
    print("Calculating distances.")
    dists = euclidean_distances(x_toexplain.to_frame().T, X)
    radius_all_ = dists.max()*numpy.array(radius_perc)
    
    for i in range(len(radius_all_)):
        radius = radius_all_[i]
        support_x_ = generate_inside_ball(numpy.array(x_toexplain), segment=(0, radius), n=1000)

        pap_fid = model_eval(sess, x, y, preds_sub, support_x_, bb_model.predict(support_x_) , args=eval_params)
        papernot[radius_perc[i]].append(pap_fid)
        ls_fid = accuracy_score(train_sub_ls.predict(support_x_), pred(support_x_))
        localsurrogate[radius_perc[i]].append(ls_fid)


out_localsurr = pandas.DataFrame(localsurrogate)
out_papernot = pandas.DataFrame(papernot)
out_localsurr.to_csv('results/tables/exp1_german_localsurr5.csv')
out_papernot.to_csv('results/tables/exp1_german_papernot5.csv')


import seaborn as sns
import matplotlib.pyplot as plt

sns.pointplot(data=out_papernot)
sns.pointplot(data=out_localsurr, color='orange')
plt.xlabel('Radius percent')
plt.ylabel('Local Accuracy')
plt.savefig('results/figures/local_fidelity_german5.pdf')

