#This file is created to load the training and test data sets and feature selection

import sys
import os
import pandas as pd


base_folder    = os.path.join(os.getcwd(), '..', '..')

#Path to sampled training and test data
training_data = (r'D:\rubenwe\Documents\MainProgramming\Data\data_vrachtwagen_training.csv')
test_data = (r'D:\rubenwe\Documents\MainProgramming\Data\data_vrachtwagen_test.csv')

#Independent factors + dependent factor (speed) to select from data set
feature_sel_SVR = ['SpdLimit', 'LengthMeter', 'IsPaved', 'HasSpdBumps', 'NrSharpTurns'
                     ,'IsPriority', 'IsUrban', 'IsRamp']
feature_sel_RF = ['SpdLimit', 'LengthMeter', 'IsPaved', 'HasSpdBumps', 'NrSharpTurns'
                     ,'IsPriority', 'IsUrban', 'IsRamp']
feature_sel_LR = ['SpdLimit', 'LengthMeter', 'IsPaved', 'HasSpdBumps', 'NrSharpTurns'
                     ,'IsPriority', 'IsUrban', 'IsRamp']                 
feature_sel_NN = ['SpdLimit', 'LengthMeter', 'IsPaved', 'HasSpdBumps', 'NrSharpTurns'
                     ,'IsPriority', 'IsUrban', 'IsRamp']
                     
                     #, 'SpdPatMax', 'SpdPatAvg', 'SpdPatMin', 'HasTrafficSignal', ] 
            
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
    











