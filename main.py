#!/usr/bin/env python3
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from random import randint
from numpy import array
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

####################################################
# PARAMETERS
####################################################

VALUE_COUNT_TRAIN = 10000
MAX_VALUE_TRAIN = 500
MAX_VALUE_TEST = 500
EPOCHS = 50
BATCH_SIZE = 128
VERBOSE = 1
ACTIVATION = "linear"
MODEL_FILE_NAME = "trained_model.h5"

####################################################

# Create Numpy ndarray triples with two numbers and sum¶
def create_triples (count, max):
    numbers = list ()
    result = list ()
    for n in range (count):
        numbers.append ([randint (0, max), randint (0, max)])
        result.append (sum (numbers[n]))
    numbers = array (numbers)
    result = array (result)
    return numbers, result

# Denormalize numbers
def denormalize (value, max):
    return value * float (max)

# Normalize number to range
def normalize (value, max):
    return value.astype ('float') / float (max)

# Model create
def model_create ():
    model = Sequential()
    model.add (Dense (3, input_dim = 2, activation = 'linear'))
    model.add (Dense (5, activation = 'linear'))
    model.add (Dense (1, activation = 'linear'))
    model.compile (loss='mean_squared_error', optimizer='adam')
    return model

# Train model
def model_train (model):
        x, y = create_triples (VALUE_COUNT_TRAIN, MAX_VALUE_TRAIN)
        x2 = normalize (x, MAX_VALUE_TRAIN)
        y2 = normalize (y, MAX_VALUE_TRAIN)
        model.fit (x2,
               y2,
               epochs = EPOCHS,
               batch_size = BATCH_SIZE,
               verbose = VERBOSE)


def save_model (m, filename):
    model.save (filename)

# Load or Create model
load_model_status = False

if load_model_status:
    print ("Load model\n")
    model = load_model (MODEL_FILE_NAME)
else:
    print ("Create model")
    model = model_create ()
    print (model.summary ())
    model_train (model)
    model.save (MODEL_FILE_NAME)
    print ("Save_model\n")

# Prediction
x, y = create_triples (10, MAX_VALUE_TEST)
x2 = normalize (x, MAX_VALUE_TEST)
testresult = model.predict (x2, verbose = 0)

# Show result
print ('Count      Number1        Number2       Total            Right           Error')
for i in range (len (testresult)):
    number1 = denormalize (x2[i][0], MAX_VALUE_TEST)
    number2 = denormalize (x2[i][1], MAX_VALUE_TEST)
    total = denormalize (testresult[i][0], MAX_VALUE_TEST)
    print ('{:4d}. {:12.3f} + {:12.3f} = {:12.3f} === {:12.3f} {:12.3f}'.format (
		i + 1, number1, number2, total, number1 + number2, abs(total - (number1 + number2)))
	)
