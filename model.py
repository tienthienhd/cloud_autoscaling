import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, GRU, Dropout, InputLayer, Input
from keras.optimizers import Adam, Nadam, Adamax, RMSprop, SGD, Adagrad, Adadelta
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, \
    EarlyStopping, ModelCheckpoint, TerminateOnNaN, TensorBoard
from keras import backend as K
from keras import activations, initializers, optimizers
from keras.layers import Layer
import tensorflow as tf
from bayesian_nn.dense_variational import DenseVariational, neg_log_likelihood


def ann_model(layer_sizes, input_shape, activation, dropout, optimizer, learning_rate):
    input = Input(shape=input_shape, name='input')
    net = input
    for i, units in enumerate(layer_sizes):
        layer = Dense(units=units, activation=activation, name='layer_%d' % i)
        net = layer(net)
        drop = Dropout(dropout, name='dropout_layer_%d' % i)
        net = drop(net, training=False)
    preds = Dense(units=1, name='output')(net)

    model = Model(input, preds)
    opt = optimizer(lr=learning_rate)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    model.summary()
    return model


# ann_model([3, 4], (3, 4), 'tanh', 0.5, Adam, 0.01)


def lstm_model(layer_sizes, input_shape, activation, dropout, recurrent_dropout, optimizer, learning_rate):
    input = Input(shape=input_shape, name='input')
    net = input
    for i, units in enumerate(layer_sizes):
        if i < len(layer_sizes) - 1:

            layer = LSTM(units=units, activation=activation, dropout=dropout,
                         recurrent_dropout=recurrent_dropout, return_sequences=True,
                         name='layer_%d' % i)
        else:
            layer = LSTM(units=units, activation=activation, dropout=dropout,
                         recurrent_dropout=recurrent_dropout, return_sequences=False,
                         name='layer_%d' % i)

        net = layer(net)
    preds = Dense(units=1, name='output')(net)

    model = Model(input, preds)
    opt = optimizer(lr=learning_rate)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    model.summary()
    return model


# lstm_model([4, 3], (5, 4), 'tanh', 0.5, 0.5, Adam, 0.001)


def ed_model(layer_sizes, input_shape_e, input_shape_d, activation, dropout, recurrent_dropout, optimizer,
             learning_rate, multi_output_decoder=True):
    state_encoder = []
    input_e = Input(shape=input_shape_e, name='input_encoder')
    net_e = input_e
    for i, units in enumerate(layer_sizes):
        layer = LSTM(units=units, activation=activation, dropout=dropout, recurrent_dropout=recurrent_dropout,
                     return_sequences=True, return_state=True, name='encoder_layer_%d' % i)
        net_e, h, c = layer(net_e, training=False)
        state_encoder.append([h, c])
    feature = net_e

    # ===============================================================
    input_d = Input(shape=input_shape_d, name='input_decoder')
    net_d = input_d
    for i, units in enumerate(layer_sizes):
        if i < len(layer_sizes) - 1:
            layer = LSTM(units=units, activation=activation, dropout=dropout, recurrent_dropout=recurrent_dropout,
                         return_sequences=True, name='decoder_layer_%d' % i)
        else:
            layer = LSTM(units=units, activation=activation, dropout=dropout, recurrent_dropout=recurrent_dropout,
                         return_sequences=multi_output_decoder, name='decoder_layer_%d' % i)
        net_d = layer(net_d, initial_state=state_encoder[i], training=False)
    d_preds = Dense(units=1, name='predict_decoder')(net_d)

    ed_model = Model([input_e, input_d], d_preds)
    optimizer = optimizers.get(optimizer)
    optimizer.lr = learning_rate
    ed_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    # ed_model.summary(line_length=200)
    return ed_model


# model = ed_model([32], [3, 4], [5, 4], 'tanh', 0.5, 0.5, 'adam', 0.01)



def lstm_dense_variational(input_shape, kl_loss_weight):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(LSTM(units=32, activation='tanh', dropout=0.05, recurrent_dropout=0.05,
                   return_sequences=True))
    model.add(LSTM(units=8, activation='tanh', dropout=0.05, recurrent_dropout=0.05,
                   return_sequences=False))
    model.add(DenseVariational(32, kl_loss_weight=kl_loss_weight, activation='tanh'))
    # model.add(DenseVariational(8, kl_loss_weight=kl_loss_weight, activation='tanh'))
    model.add(DenseVariational(1, kl_loss_weight=kl_loss_weight))

    model.compile(loss=neg_log_likelihood, optimizer='adam', metrics=['mse', 'mae'])
    return model


