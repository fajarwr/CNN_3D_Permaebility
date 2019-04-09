from __future__ import division, print_function, absolute_import
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D, BatchNormalization, Input
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, TensorBoard
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score
import os
import sys
import matplotlib.pyplot as plt
import re
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.callbacks import ModelCheckpoint

#Define r square matric
def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

#Change to script directory
os.chdir(sys.path[0])
sys.path.append(os.getcwd())
#Import datagenerator taken from 
#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
from DataGenerator_3D_Classes import DataGenerator

#Experiment number
exp_num = 16
os.chdir('..\\..\\005_Result\\CNN_3D')
if any('00'+str(exp_num) in s for s in os.listdir(os.getcwd())):
    sys.exit('Alert : There is already 00'+str(exp_num)+' experiment result!!')

#Load the data
dim1,dim2,dim3,chn = 100,100,100,1
training_len = 90
testing_len = 10
total_len = training_len + testing_len
phi = []
ssa = []
os.chdir('..\\..\\002_Data\\Berea_Sandstone_npy')
for image3D_npy in os.listdir(os.getcwd())[:total_len]:
    phi.append([float(s) for s in re.findall('[-+]?\d*\.\d+|\d+',
                image3D_npy)][1])
    ssa.append([float(s) for s in re.findall('[-+]?\d*\.\d+|\d+',
                image3D_npy)][2])
k = np.power(1-np.array(phi), 3)/np.power(ssa, 2)
k_norm = k/np.max(k)

#Plot the data
#plt.scatter(np.arange(1,9262),k[:9261], s = 3)
#
#plt.scatter((1-np.array(phi)),k)
#plt.yscale('log')

# Parameters
params = {'dim': (dim1,dim2,dim3),
          'batch_size': 20,
          'n_classes': 1,
          'n_channels': chn,
          'shuffle': False}

#Datasets
partition = {
		'train': os.listdir(os.getcwd())[:training_len],
		'validation': os.listdir(os.getcwd())[training_len:total_len],
        'total' : os.listdir(os.getcwd())[:total_len]
		}
labels = dict(zip(os.listdir(os.getcwd())[:total_len], k_norm))

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)
total_generator = DataGenerator(partition['total'], labels, **params)

#Define a model
model = Sequential()
model.add(Conv3D(16, kernel_size=5, strides=(2, 2, 2), padding='valid',
                 data_format='channels_last', dilation_rate=(1, 1, 1),
                 activation='relu', use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, 
                 activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, input_shape=(dim1, dim2, dim3, chn)))
model.add(Conv3D(16, kernel_size=3, strides=(1, 1, 1), padding='valid',
                 data_format='channels_last', dilation_rate=(1, 1, 1),
                 activation='relu', use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, 
                 activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='valid',
                       data_format='channels_last'))
model.add(Conv3D(16, kernel_size=3, strides=(1, 1, 1), padding='valid',
                 data_format='channels_last', dilation_rate=(1, 1, 1),
                 activation='relu', use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, 
                 activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None))
model.add(Conv3D(16, kernel_size=3, strides=(1, 1, 1), padding='valid',
                 data_format='channels_last', dilation_rate=(1, 1, 1),
                 activation='relu', use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, 
                 activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='valid',
                       data_format='channels_last'))
model.add(Flatten(data_format='channels_last'))
model.add(Dense(128, activation='relu', use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=None, bias_regularizer=None,
                activity_regularizer=None, kernel_constraint=None,
                bias_constraint=None))
model.add(Dense(128, activation='relu', use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=None, bias_regularizer=None,
                activity_regularizer=None, kernel_constraint=None,
                bias_constraint=None))
model.add(Dense(64, activation='relu', use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=None, bias_regularizer=None,
                activity_regularizer=None, kernel_constraint=None,
                bias_constraint=None))
model.add(Dense(1, activation=None, use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=None, bias_regularizer=None,
                activity_regularizer=None, kernel_constraint=None,
                bias_constraint=None))

#Compile the model
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=[r2_keras],
              loss_weights=None, sample_weight_mode=None,weighted_metrics=None,
              target_tensors=None)

#This checkpoint object will store the model parameters in the file "weights.hdf5"
checkpoint = ModelCheckpoint(filepath='..\\..\\005_Result\\CNN_3D\\Weights_CNN3D_00'+str(exp_num)+'.hdf5', monitor='val_loss')

#Change to data directory
os.chdir(sys.path[0])
os.chdir('..\\..\\002_Data\\Berea_Sandstone_npy')

# Train model on dataset
history = model.fit_generator(generator=training_generator, epochs=20,
                    callbacks=[checkpoint], workers=1, use_multiprocessing=False)

#Save history
history_df = pd.DataFrame.from_dict(history.history)
history_df.to_excel('..\\..\\005_Result\\CNN_3D\\History_CNN3D_00'+str(exp_num)+'.xlsx')

#Load the model and plot the data
model.load_weights('..\\..\\005_Result\\CNN_3D\\Weights_CNN3D_00'+str(exp_num)+'.hdf5')

#Store the training & testing result
total_result = model.predict_generator(generator=total_generator, steps=None,
                                  max_queue_size=10, workers=1,
                                  use_multiprocessing=False, verbose=0)

#Save result
training_result = {
		'true_training': np.reshape(k_norm[:training_len]*np.max(k),(training_len,)),
		'pred_training': np.reshape(total_result[:training_len]*np.max(k),(training_len,))
		}
testing_result = {
        'true_testing': np.reshape(k_norm[training_len:total_len]*np.max(k),(testing_len,)),
		'pred_testing': np.reshape(total_result[training_len:total_len]*np.max(k),(testing_len,))
        }
training_result_df = pd.DataFrame.from_dict(training_result)
testing_result_df = pd.DataFrame.from_dict(testing_result)
training_result_df.to_excel('..\\..\\005_Result\\CNN_3D\\Training_CNN3D_00'+
                            str(exp_num)+'_'+str(r2_score (k_norm[:training_len]*np.max(k), total_result[:training_len]*np.max(k)))+'.xlsx')
testing_result_df.to_excel('..\\..\\005_Result\\CNN_3D\\Testing_CNN3D_00'+
                           str(exp_num)+'_'+str(r2_score(k_norm[training_len:total_len]*np.max(k), total_result[training_len:total_len]*np.max(k)))+'.xlsx')


#Plot the training data
plt.figure()
plt.scatter(np.arange(0,training_len),k_norm[:training_len]*np.max(k), label='$\kappa$ true')
plt.scatter(np.arange(0,training_len),total_result[:training_len]*np.max(k), label='$\kappa$ pred')
plt.title('Permeabilitas Kozeny Carman vs CNN Data Training')
plt.xlabel('Subampel')
plt.ylabel('$\phi^3/ssa^2$')
plt.legend()

#Plot the testing data
plt.figure()
plt.scatter(np.arange(0,testing_len),k_norm[training_len:total_len]*np.max(k), label='$\kappa$ true')
plt.scatter(np.arange(0,testing_len),total_result[training_len:total_len]*np.max(k), label='$\kappa$ pred')
plt.title('Permeabilitas Kozeny Carman vs CNN Data Testing')
plt.xlabel('Subsampel')
plt.ylabel('$\phi^3/ssa^2$')
plt.legend()

#PLot history MSE
plt.figure()
plt.plot(history.history['loss'])
plt.title('PLot Nilai Mean Square Error untuk Setiap Epoch')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()

#Plot history r2_keras
plt.figure()
plt.plot(history.history['r2_keras'])
plt.title('PLot Nilai $R^2$ untuk Setiap Epoch')
plt.xlabel('Epoch')
plt.ylabel('$R^2$')
plt.show()


