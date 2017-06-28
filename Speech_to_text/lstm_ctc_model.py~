#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import scipy.io.wavfile as wav
import numpy as np
from six.moves import xrange as range
from python_speech_features import mfcc
from utils import sparse_tuple_from as sparse_tuple_from
from utils import pad_sequences as pad_sequences

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#configuring memory usage for tensorflow
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.operation_timeout_in_ms = 15000   # terminate on long hangs

audio_filename = 'p226_003.wav'
fs, audio = wav.read(audio_filename)
inputs = mfcc(audio, samplerate=fs)
# Tranform in 3D array
sample_label = np.asarray(inputs[np.newaxis, :])
sample_label, _ = pad_sequences(sample_label)
sample_label = (sample_label - np.mean(sample_label))/np.std(sample_label)


# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

# Some configs
num_features = 13
# Accounting the 0th indice +  space + blank label = 28 characters
num_classes = ord('z') - ord('a') + 1 + 1 + 1

# Hyper-parameters
num_epochs = 1000
num_hidden = 80
num_layers = 1
batch_size = 100
initial_learning_rate = 0.00001
momentum = 0.9
num_examples = 44085  #8233
num_batches_per_epoch = int(num_examples/batch_size)

inputs = np.load('allset/train_input_all.npy')
labels = np.load('allset/train_label_all.npy')

# You can preprocess the input data here
train_inputs = inputs
train_inputs, _ = pad_sequences(train_inputs)
train_inputs = (train_inputs - np.mean(train_inputs))/np.std(train_inputs)

# You can preprocess the target data here
train_targets = labels

# THE MAIN CODE!
#We start building the model(graph) of our neural netowrk
graph = tf.Graph()
with graph.as_default():
    # e.g: log filter bank or MFCC features
    # Has size [batch_size, max_stepsize, num_features], but the
    # batch_size and max_stepsize can vary along each step
    inputs = tf.placeholder(tf.float32, [None, None, num_features])

    # Here we use sparse_placeholder that will generate a
    # SparseTensor required by ctc_loss op.
    targets = tf.sparse_placeholder(tf.int32)

    # 1d array of size [batch_size]
    seq_len = tf.placeholder(tf.int32, [None])

    # Defining the cell
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
    #cells are the layers that we will stack to form our neural network
    cell1 = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    cell1 = tf.nn.rnn_cell.DropoutWrapper(cell=cell1, output_keep_prob=0.5)
    cell2 = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    cell2 = tf.nn.rnn_cell.DropoutWrapper(cell=cell2, output_keep_prob=0.3)
    cell3 = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    cell3 = tf.nn.rnn_cell.DropoutWrapper(cell=cell3, output_keep_prob=0.2)
    cell4 = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    cell4 = tf.nn.rnn_cell.DropoutWrapper(cell=cell4, output_keep_prob=0.1)
    cell5 = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    cell6 = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    cell7 = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    cell8 = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    cell9 = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    cell10 = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    cell11 = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    cell12 = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)

    # Stacking rnn cells
    stack = tf.contrib.rnn.MultiRNNCell([cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8, cell9, cell10,
                                         cell11, cell12], state_is_tuple=True)

    # The second output is the last state and we will no use that
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]

    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, num_hidden])

    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
    # Zero initialization
    # Tip: Is tf.zeros_initializer the same?
    b = tf.Variable(tf.constant(0., shape=[num_classes]))

    # Doing the affine projection
    logits = tf.matmul(outputs, W) + b

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))

    loss = tf.nn.ctc_loss(targets, logits, seq_len)
    cost = tf.reduce_mean(loss)

    optimizer = tf.train.MomentumOptimizer(initial_learning_rate, momentum).minimize(cost)
    #optimizer = tf.train.AdamOptimizer(initial_learning_rate).minimize(cost)

    # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

    # Inaccuracy: label error rate
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

#running the graph after building it completely (feeding the actual data in the model)
print('started!')
with tf.Session(graph=graph) as session:
    # Initializate the weights and biases
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()

    for curr_epoch in range(num_epochs):
        train_cost = train_ler = 0
        start = time.time()

        for batch in range(num_batches_per_epoch):

            # Getting the index
            indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]
            #print(indexes[0])
            batch_train_inputs = train_inputs[indexes]
            # Padding input to max_time_step of this batch
            batch_train_inputs, batch_train_seq_len = pad_sequences(batch_train_inputs)

            # Converting to sparse representation so as to to feed SparseTensor input
            batch_train_targets = sparse_tuple_from(train_targets[indexes])

            feed = {inputs: batch_train_inputs,
                    targets: batch_train_targets,
                    seq_len: batch_train_seq_len}

            batch_cost, _ = session.run([cost, optimizer], feed)
            train_cost += batch_cost*batch_size
            train_ler += session.run(ler, feed_dict=feed)*batch_size

        batch_train_inputs, batch_train_seq_len = pad_sequences(sample_label)

        feed = {inputs: batch_train_inputs,
                seq_len: batch_train_seq_len}

        # Decoding
        d = session.run(decoded[0], feed_dict=feed)
        str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
        # Replacing blank label to none
        str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
        # Replacing space label to space
        str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
        print('Decoded_inter:\n%s' % str_decoded)

        # Shuffle the data
        shuffled_indexes = np.random.permutation(num_examples)
        train_inputs = train_inputs[shuffled_indexes]
        train_targets = train_targets[shuffled_indexes]

        # Metrics mean
        train_cost /= num_examples
        train_ler /= num_examples

        log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}"
        print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler, time.time() - start))
        #saving the model every 5th epoch
        if((curr_epoch+1)%5 == 0 ):
            print("model saving starts")
            saver.save(session, 'saved_models/final_model/s2t')
            print("model is saved")


    print("model saving starts")
    saver.save(session, 'saved_models/final_model/s2t')
    print("model is saved")

    batch_train_inputs, batch_train_seq_len = pad_sequences(sample_label)

    feed = {inputs: batch_train_inputs,
            seq_len: batch_train_seq_len}

    # Decoding
    d = session.run(decoded[0], feed_dict=feed)
    str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
    # Replacing blank label to none
    str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
    # Replacing space label to space
    str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
    print('Decoded:\n%s' % str_decoded)
