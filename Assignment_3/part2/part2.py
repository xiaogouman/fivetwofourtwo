from __future__ import print_function
import numpy as np
import math, os
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, Flatten

# Create model
def create_rnn_model(stateful,length):
	##### YOUR MODEL GOES HERE #####
	model = Sequential()
	model.add(SimpleRNN(units=20,activation='relu',stateful=stateful,batch_input_shape=(1, 1, length)))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mean_squared_error')
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
	x_train = np.array(x_train)
	x_test = np.array(x_test)
	x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
	x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
	return (x_train, np.array(y_train)), (x_test, np.array(y_test))

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
rnn_stateful_rmse_list=list()
rnn_stateless_rmse_list=list()

# result directory
directory = '/Users/xiaogouman/Documents/masters/CS5242/Assignments/Assignment_3/results/part2/'
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
	
	# Create the stateful model
	print('Creating Stateful Vanilla RNN Model...')
	model_rnn_stateful = create_rnn_model(stateful=True,length=length)

	# cost of stateful model after each epoch
	rnn_stateful_train_loss = []
	rnn_stateful_test_loss = []

	# Train the model
	print('Training')
	for i in range(epochs):
		print('Epoch', i + 1, '/', epochs)
		# Note that the last state for sample i in a batch will
		# be used as initial state for sample i in the next batch.
		
		##### TRAIN YOUR MODEL #####
		h = model_rnn_stateful.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=2, shuffle=False,
								   validation_data=(x_test, y_test))

		loss, val_loss = h.history['loss'], h.history['val_loss']
		rnn_stateful_train_loss.append(loss)
		rnn_stateful_test_loss.append(val_loss)

		# reset states at the end of each epoch
		model_rnn_stateful.reset_states()


	# Plot and save loss curves of training and test set vs iteration in the same graph
	##### PLOT AND SAVE LOSS CURVES #####
	epoch_num = np.linspace(1, epochs, num=epochs)
	plt.figure(1)
	plt.plot(epoch_num, rnn_stateful_train_loss, 'r-', linewidth=2, label='loss of train set')
	plt.plot(epoch_num, rnn_stateful_test_loss, 'b-', linewidth=2, label='loss of test set')
	plt.xlabel('epoch')
	plt.ylabel('loss', fontsize=15)
	plt.title('Decrease of Loss, Vanilla RNN, Stateful')
	plt.legend()
	plt.grid()
	plt.savefig(directory + 'model_rnn_stateful_weights_length_' + str(length) + '.png')
	plt.close()

	# Save your model weights with following convention:
	# For example length 1 input sequences model filename
	# rnn_stateful_model_weights_length_1.h5
	##### SAVE MODEL WEIGHTS #####
	filename = directory+\
		'model_rnn_stateful_weights_length_'\
		+str(length)
	model_rnn_stateful.save_weights(filename)

	# Predict 
	print('Predicting')
	##### PREDICT #####
	predicted_rnn_stateful = model_rnn_stateful.predict(x_test, batch_size=batch_size)

	##### CALCULATE RMSE #####
	rnn_stateful_rmse = math.sqrt(sum((predicted_rnn_stateful - y_test)**2)/len(y_test))
	rnn_stateful_rmse_list.append(rnn_stateful_rmse)

	# print('tsteps:{}'.format(tsteps))
	print('length:{}'.format(length))
	print('Stateful Vanilla RNN RMSE:{}'.format( rnn_stateful_rmse ))



	# Create the stateless model
	print('Creating stateless Vanilla RNN Model...')
	model_rnn_stateless = create_rnn_model(stateful=False, length=length)

	# Train the model
	print('Training')
	##### TRAIN YOUR MODEL #####
	h = model_rnn_stateless.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2,
							   validation_data=(x_test, y_test), shuffle=False)


	# Plot and save loss curves of training and test set vs iteration in the same graph
	##### PLOT AND SAVE LOSS CURVES #####
	loss, val_loss = h.history['loss'], h.history['val_loss']
	epoch_num = np.linspace(1, epochs, num=epochs)
	plt.figure(1)
	plt.plot(epoch_num, loss, 'r-', linewidth=2, label='loss of train set')
	plt.plot(epoch_num, val_loss, 'b-', linewidth=2, label='loss of test set')
	plt.xlabel('epoch')
	plt.ylabel('loss', fontsize=15)
	plt.title('Decrease of Loss, Vanilla RNN, Stateless')
	plt.legend()
	plt.grid()
	plt.savefig(directory + 'model_rnn_stateless_weights_length_' + str(length) + '.png')
	plt.close()

	# Save your model weights with following convention:
	# For example length 1 input sequences model filename
	# rnn_stateless_model_weights_length_1.h5
	##### SAVE MODEL WEIGHTS #####
	filename = directory+\
		'model_rnn_stateless_weights_length_'\
		+str(length)
	model_rnn_stateless.save_weights(filename)

	# Predict 
	print('Predicting')
	##### PREDICT #####
	predicted_rnn_stateless = model_rnn_stateless.predict(x_test)

	##### CALCULATE RMSE #####
	rnn_stateless_rmse = math.sqrt(sum((predicted_rnn_stateless - y_test)**2)/len(y_test))
	rnn_stateless_rmse_list.append(rnn_stateless_rmse)

	# print('tsteps:{}'.format(tsteps))
	print('length:{}'.format(length))
	print('Stateless Vanilla RNN RMSE:{}'.format( rnn_stateless_rmse ))


# save your rmse values for different length input sequence models - stateful rnn:
filename = 'rnn_stateful_model_rmse_values.txt'
np.savetxt(filename, np.array(rnn_stateful_rmse_list), fmt='%.6f', delimiter='\t')

# save your rmse values for different length input sequence models - stateless rnn:
filename = 'rnn_stateless_model_rmse_values.txt'
np.savetxt(filename, np.array(rnn_stateless_rmse_list), fmt='%.6f', delimiter='\t')

print("#" * 33)
print('Plotting Results')
print("#" * 33)

# Plot and save rmse vs Input Length
plt.figure()
plt.plot( np.arange(min_length,max_length+1), rnn_stateful_rmse_list, c='blue', label='Stateful RNN')
plt.plot( np.arange(min_length,max_length+1), rnn_stateless_rmse_list, c='cyan', label='Stateless RNN')
plt.title('RMSE vs Input Length in Test Set')
plt.xlabel('length of input sequences')
plt.ylabel('rmse')
plt.legend()
plt.grid()
plt.savefig('rnn_model_rmse_length_'+str(max_length)+'.png')
plt.show()


