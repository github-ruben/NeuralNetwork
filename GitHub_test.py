import sys
import os
import pandas as pd


feature_sel_NN = ['SpdLimit', 'LengthMeter', 'IsPaved', 'HasSpdBumps', 'NrSharpTurns'
                     ,'IsPriority', 'IsUrban', 'IsRamp']

training_data = (r'data_vrachtwagen_training.csv')
test_data = (r'data_vrachtwagen_test.csv')

def load_and_prepare(data_file_name):
    #Load data file with speeds and factors for road segments 
    data_set_learning = pd.read_csv(data_file_name , sep = ';')
    data_set_learning = data_cleaning(data_set_learning)
    return data_set_learning

def data_cleaning(data):
    #Ignore speed if GPS point is more than 20m away
    data = data[data.MeterAwayFromGps <= 20]
    #Ignore is speed is on ferry edge  
    data = data[data.IsFerry == 0]
    #Ignore speed limits of >130
    data = data[data.SpdLimit <= 130]
    return data
    
training_data = load_and_prepare(training_data)
test_data     = load_and_prepare(test_data)

#----- Setup independent and depedent (target) variables ----------------------------------------------------------------------------------------
features      = feature_sel_NN
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
from sklearn.model_selection import cross_val_score


#from keras.layers import Dropout

# Building the Neural Network
def build_regressor(nr_neurons, nr_hiddenlayers, nr_features, optimizer = 'adam'):   
    #reproduce()
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
'''
filepath = r"PointBasedMethods\ModelCheckPoint\Best_NN_model-{val_loss:.4f}.hdf5"

#Remove files from modelcheckpoint --> otherwise error occurs    
for c in os.listdir(ModelCheckPoint_folder):
    full_path = os.path.join(ModelCheckPoint_folder, c)
    if os.path.isfile(full_path):
        os.remove(full_path)
    else:
        shutil.rmtree(full_path)
'''


regressor = build_regressor(nr_neurons, nr_hiddenlayers, nr_features)
#regressor = build_regressor(15, 8, nr_features)
#callbacks = [EarlyStopping(monitor='val_loss', patience=5), ModelCheckpoint(filepath ,mode= 'min',monitor = 'val_loss', save_best_only = 'True', verbose=1)]
regressor.fit(X_train, y_train, batch_size = batch_size, nb_epoch = nr_epochs, verbose = 2)#callbacks = callbacks, validation_data = (X_test, y_test), verbose =2) 
