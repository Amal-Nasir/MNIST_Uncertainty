import pandas as pd
import numpy as np
import csv

train_data2 = pd.read_csv('/home/aalmansour/source/lidc_slices/MNIST/four_raters/high_train_data.csv') 
validation_data2 = pd.read_csv('/home/aalmansour/source/lidc_slices/MNIST/four_raters/high_validation_data.csv')
test_data2 = pd.read_csv('/home/aalmansour/source/lidc_slices/MNIST/four_raters/high_test_data.csv')
print('train high data len:', len(train_data2))
print('val high data len:', len(validation_data2))
print('test high data len:', len(test_data2))
print('----------------------------------------------')

train_data3 = pd.read_csv('/home/aalmansour/source/lidc_slices/MNIST/four_raters/low_train_data.csv') 
validation_data3 = pd.read_csv('/home/aalmansour/source/lidc_slices/MNIST/four_raters/low_validation_data.csv')
test_data3 = pd.read_csv('/home/aalmansour/source/lidc_slices/MNIST/four_raters/low_test_data.csv')
print('train low data len:', len(train_data3))
print('val low data len:', len(validation_data3))
print('test low data len:', len(test_data3))
print('----------------------------------------------')

train_data4 = pd.read_csv('/home/aalmansour/source/lidc_slices/MNIST/four_raters/no_train_data.csv')
validation_data4 = pd.read_csv('/home/aalmansour/source/lidc_slices/MNIST/four_raters/no_validation_data.csv')
test_data4 = pd.read_csv('/home/aalmansour/source/lidc_slices/MNIST/four_raters/no_test_data.csv')
print('train no data len:', len(train_data4))
print('val no data len:', len(validation_data4))
print('test no data len:', len(test_data4))
print('----------------------------------------------')