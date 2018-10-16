from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN

# Create model
def create_fc_model():
	##### YOUR MODEL GOES HERE #####
	model = Sequential()
	model.add(Dense(20, activation='relu'))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# split train/test data
def split_data(x, y, ratio=0.8):
	to_train = int(len(x.index) * ratio)
	# tweak to match with batch_size
	to_train -= to_train % batch_size

	x_train = x[:to_train]
	y_train = y[:to_train]
	x_test = x[to_train:]
	y_test = y[to_train:]

	# tweak to match with batch_size
	to_drop = x.shape[0] % batch_size
	if to_drop > 0:
		x_test = x_test[:-1 * to_drop]
		y_test = y_test[:-1 * to_drop]

	# some reshaping
	##### RESHAPE YOUR DATA BASED ON YOUR MODEL #####
	return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))

# training parameters passed to "model.fit(...)"
batch_size = 1
epochs = 10

# The input sequence min and max length that the model is trained on for each output point
min_length = 1
max_length = 10

# load data from files
noisy_data = np.loadtxt('../filter_data/noisy_data.txt', delimiter='\t', dtype=np.float)
smooth_data = np.loadtxt('../filter_data/smooth_data.txt', delimiter='\t', dtype=np.float)

print('noisy_data shape:{}'.format(noisy_data.shape))
print('smooth_data shape:{}'.format(smooth_data.shape))
print('noisy_data first 5 data points:{}'.format(noisy_data[:5]))
print('smooth_data first 5 data points:{}'.format(smooth_data[:5]))


# List to keep track of root mean square error for different length input sequences
fc_rmse_list=list()

# result directory
directory = '/Users/xiaogouman/Documents/masters/CS5242/Assignments/Assignment_3/results/part1/'
if not os.path.exists(directory):
	os.makedirs(directory)

for num_input in range(min_length,max_length+1):
	length = num_input

	print("*" * 33)
	print("INPUT DIMENSION:{}".format(length))
	print("*" * 33)

	# convert numpy arrays to pandas dataframe
	data_input = pd.DataFrame(noisy_data)
	expected_output = pd.DataFrame(smooth_data)

	# when length > 1, arrange input sequences
	if length > 1:
		##### ARRANGE YOUR DATA SEQUENCES #####
		temp_input = data_input
		for i in range(1, length):
			data_input = pd.concat([data_input, temp_input.shift(i)], axis=1)
		data_input = data_input.fillna(0)

	print('data_input length:{}'.format(len(data_input.index)) )

	# Split training and test data: use first 80% of data points as training and remaining as test
	(x_train, y_train), (x_test, y_test) = split_data(data_input, expected_output)
	print('x_train.shape: ', x_train.shape)
	print('y_train.shape: ', y_train.shape)
	print('x_test.shape: ', x_test.shape)
	print('y_test.shape: ', y_test.shape)

	print('Input shape:', data_input.shape)
	print('Output shape:', expected_output.shape)
	print('Input head: ')
	print(data_input.head())
	print('Output head: ')
	print(expected_output.head())
	print('Input tail: ')
	print(data_input.tail())
	print('Output tail: ')
	print(expected_output.tail())
	
	# Create the model
	print('Creating Fully-Connected Model...')
	model_fc = create_fc_model()

	# Train the model
	print('Training')
	##### TRAIN YOUR MODEL #####
	h = model_fc.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=(x_test, y_test))

	# Plot and save loss curves of training and test set vs iteration in the same graph
	##### PLOT AND SAVE LOSS CURVES #####
	loss, val_loss = h.history['loss'], h.history['val_loss']
	epoch_num = np.linspace(1, epochs, num=epochs)
	plt.figure(1)
	plt.plot(epoch_num, loss, 'r-', linewidth=2, label='loss of train set')
	plt.plot(epoch_num, val_loss, 'b-', linewidth=2, label='loss of test set')
	plt.xlabel('epoch')
	plt.ylabel('loss', fontsize=15)
	plt.title('Decrease of Loss')
	plt.legend()
	plt.grid()
	plt.savefig(directory + 'fc_model_weights_length_'+str(length)+'.png')
	plt.close()

	# Save your model weights with following convention:
	# For example length 1 input sequences model filename
	# fc_model_weights_length_1.h5
	##### SAVE MODEL WEIGHTS #####

	filename = \
		'/Users/xiaogouman/Documents/masters/CS5242/Assignments/Assignment_3/results/part1/fc_model_weights_length_'\
		+str(length)
	model_fc.save_weights(filename)

	# Predict 
	print('Predicting')
	##### PREDICT #####
	predicted_fc = model_fc.predict(x_test)

	##### CALCULATE RMSE #####
	fc_rmse = math.sqrt(sum((predicted_fc - y_test)**2)/len(y_test))
	fc_rmse_list.append(fc_rmse)

	# print('tsteps:{}'.format(tsteps))
	print('length:{}'.format(length))
	print('Fully-Connected RMSE:{}'.format( fc_rmse ))

# save your rmse values for different length input sequence models:
filename = 'fc_model_rmse_values.txt'
np.savetxt(filename, np.array(fc_rmse_list), fmt='%.6f', delimiter='\t')

print("#" * 33)
print('Plotting Results')
print("#" * 33)

# Plot and save rmse vs Input Length
plt.figure()
plt.plot( np.arange(min_length,max_length+1), fc_rmse_list, c='black', label='FC')
plt.title('RMSE vs Input Length in Test Set')
plt.xlabel('length of input sequences')
plt.ylabel('rmse')
plt.legend()
plt.grid()
plt.savefig(directory + 'rmse_length'+str(length)+'.png')
plt.show()
plt.close()


