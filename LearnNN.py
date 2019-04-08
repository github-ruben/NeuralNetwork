def reproduce():
    #As mentioned in Keras FAQ add the following code:
    import numpy as np
    import tensorflow as tf
    import random as rn
    
    # The below is necessary in Python 3.2.3 onwards to
    # have reproducible behavior for certain hash-based operations.
    # See these references for further details:
    # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
    # https://github.com/keras-team/keras/issues/2280#issuecomment-306959926
    
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    
    np.random.seed(4)
    
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    
    rn.seed(15)
    
    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of
    # non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds have-to-be-set-where-to-realize-100-reproducibility-of-training-res
    
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, 
    inter_op_parallelism_threads=1)
    
    from keras import backend as K
    
    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    
    tf.set_random_seed(26)
    
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    return

reproduce()

#----- Imports--------------------------------------------------------------------------------------------------------------------------
import sys
import os
import glob

import numpy as np
import pickle

#Import Python Files
import LoadAndPrepare as lap
os.chdir(r'D:\rubenwe\Documents\MainProgramming')
from ErrorFunctions import ErrorFunctions

#----- Setup-----------------------------------------------------------------------------------------------------------------------------
base_folder = os.path.join(os.getcwd(), )
TrainedModels_folder = os.path.join(base_folder, 'TrainedModels')
FeatureScaling_folder = os.path.join(base_folder, 'FeatureScaling')
ModelCheckPoint_folder = os.path.join(base_folder, 'PointBasedMethods\ModelCheckPoint')
file_name = 'final_model_NN.pkl'


#def LearnNN():
#----- Read training and test data --------------------------------------------------------------------------------------------------
training_data = lap.load_and_prepare(lap.training_data)
test_data     = lap.load_and_prepare(lap.test_data)

#----- Setup independent and depedent (target) variables ----------------------------------------------------------------------------------------
features      = lap.feature_sel_LR
nr_features   = len(features)

X_train = training_data[features]
X_test  = test_data[features]

y_train = training_data.CrwlKmph
y_test  = test_data.CrwlKmph

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#----- Building NN -------------------------------------------------------------------------------------------
# Importing the Keras libraries and packages
import keras
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import cross_val_score


#from keras.layers import Dropout

# Building the Neural Network
def build_regressor(nr_neurons, nr_hiddenlayers, nr_features, optimizer = 'adam'):   
    reproduce()
    regressor = Sequential()
    
    # Adding the input layer and the first hidden layer
    regressor.add(Dense(units = nr_neurons, kernel_initializer = 'uniform', activation = 'relu', input_dim = nr_features))
    
    for layers in range(nr_hiddenlayers-1):
        # Adding (nr_hiddenlayers - 1) to NN
        regressor.add(Dense(units = nr_neurons, kernel_initializer = 'uniform', activation = 'relu'))

    # Adding the output layer
    regressor.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))
    
    # Compiling the ANN
    regressor.compile(optimizer = optimizer, loss = 'mean_squared_error')
    #print(regressor.get_weights())
    return regressor

#----- Learning --------------------------------------------------------------------------------------------------------------------------
'''
def run(x):
    nr_neuron = int(x[0,0])
    nr_hiddenlayer = int(x[0,1])
    epoch = int(x[0,2])
    batch_size = int(x[0,3])  
    print(nr_neurons, nr_hiddenlayers,epoch,batch_size)
    regressor = build_regressor(nr_neuron, nr_hiddenlayer)
    callbacks = [EarlyStopping(monitor='val_loss', patience=2)]
    regressor.fit(X_train, y_train, batch_size = batch_size, nb_epoch = epoch, callbacks = callbacks, validation_data = (X_test, y_test))
    return regressor.evaluate(X_test, y_test)
'''

 #---- Hyperparameter tuning ----------------------------------------- https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
#Common hyperparameters for Neural Networks, from most important to least important, are:
#Learning rate – α (is also adam parameter)
#Number of hidden units for different layers
#Momentum – β ()
#Mini-batch size
#Number of hidden layers
#Adam’s hyperparameter – β1, β2, ε (lr=0.001,beta_1=0.9,beta_2 = 0.999)
#Learning rate decay

import GPyOpt
import GPy
from GPy import kern
from GPy import models
#from GPy import priors
'''
def hyp_optimizing(x):
    nr_neurons = int(x[0,0])
    nr_hiddenlayers = int(x[0,1])
    epoch = int(x[0,2])
    batch_size = int(x[0,3])
    print(nr_neurons, nr_hiddenlayers, epoch, batch_size)
    
    #regressor = build_regressor(nr_neurons, nr_hiddenlayers)
    #callbacks = [EarlyStopping(monitor='val_loss', patience=2), ModelCheckpoint(filepath = 'best_model_NN',monitor = 'val-loss', save_best_only = 'True')]

    neural_network = KerasRegressor(build_fn = build_regressor, 
                                    batch_size= batch_size, 
                                    epochs = epoch, 
                                    nr_neurons = nr_neurons,
                                    nr_hiddenlayers = nr_hiddenlayers,
                                    nr_features = nr_features,
                                    verbose = 2)
    cv_result = cross_val_score(estimator = neural_network, 
                                X = X_train, y = y_train, 
                                cv=5, verbose =3,
                                scoring = 'mean_squared_error',
                                fit_params = {'callbacks':[EarlyStopping(monitor='val_loss', patience=5),
                                ModelCheckpoint(filepath = 'best_model_NN',monitor = 'val-loss', save_best_only = 'True')], 
                                              'validation_data' : (X_test, y_test)})
    loss = cv_result.mean() #regressor.evaluate(X_test, y_test)
    print ('The loss of the CV = ', loss)
    print ('')
    return loss#regressor.evaluate(X_test, y_test)

mixed_domain =[{'name': 'nr_neurons'     , 'type': 'discrete', 'domain': [x for x in range(6,20)]    },
               {'name': 'nr_hiddenlayers', 'type': 'discrete', 'domain': [x for x in range(1,5)]     },
               {'name': 'epochs'         , 'type': 'discrete', 'domain': [x for x in range(2,3)]   },
               {'name': 'batch_size'     , 'type': 'discrete', 'domain': [x for x in range(150,200)] }]


myProblem = GPyOpt.methods.BayesianOptimization(f = hyp_optimizing,
                                                domain = mixed_domain,                                                 
                                                initial_design_numdata = 2,
                                                acquisition_type = 'EI_MCMC',
                                                model_type = 'GP_MCMC', 
                                                maximize = True)
                                                
myProblem.run_optimization(1)
myProblem.plot_convergence()
myProblem.plot_acquisition()

#------ Train best model ------------------------------------------------------------------------------------------
print(myProblem.x_opt, myProblem.fx_opt)
hyper_train = myProblem.x_opt

nr_neurons, nr_hiddenlayers, nr_epochs, batch_size = int(hyper_train[0]), int(hyper_train[1]), int(hyper_train[2]), int(hyper_train[3])
'''
nr_neurons, nr_hiddenlayers, nr_epochs, batch_size = 20,10,100, 150

filepath = r"PointBasedMethods\ModelCheckPoint\Best_NN_model-{val_loss:.4f}.hdf5"

#Remove files from modelcheckpoint --> otherwise error occurs    
for c in os.listdir(ModelCheckPoint_folder):
    full_path = os.path.join(ModelCheckPoint_folder, c)
    if os.path.isfile(full_path):
        os.remove(full_path)
    else:
        shutil.rmtree(full_path)



regressor = build_regressor(nr_neurons, nr_hiddenlayers, nr_features)
#regressor = build_regressor(15, 8, nr_features)
callbacks = [EarlyStopping(monitor='val_loss', patience=5), ModelCheckpoint(filepath ,mode= 'min',monitor = 'val_loss', save_best_only = 'True', verbose=1)]
regressor.fit(X_train, y_train, batch_size = batch_size, nb_epoch = nr_epochs, callbacks = callbacks, validation_data = (X_test, y_test), verbose =2) 

#prints solution of trained model of last epoch
print(regressor.evaluate(X_test, y_test))

#Load best trained model and print val_loss of best model
files_path = os.path.join(r"D:\rubenwe\Documents\MainProgramming\PointBasedMethods\ModelCheckPoint", '*')
files = sorted(
    glob.iglob(files_path), key=os.path.getctime, reverse=True) 
regressor.load_weights(files[0])
print(regressor.evaluate(X_test, y_test))


#----- Prediction ------------------------------------------------------------------------------------------------
y_pred = regressor.predict(X_test)
ErrorFunctions(y_test,y_pred)


#----- Save Final Model ------------------------------------------------------------------------------------------------
final_model = open(os.path.join(TrainedModels_folder, file_name), 'wb')
pickle.dump(regressor, final_model)
final_model.close()


#----- Save Feature Scaling function ----------------------------------------------------------------------------
feature_sc = open(os.path.join(FeatureScaling_folder, 'feature_scaling_NN'), 'wb')
pickle.dump(sc, feature_sc)
feature_sc.close()
#    return


