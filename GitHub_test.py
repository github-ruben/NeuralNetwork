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