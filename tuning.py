import matplotlib.pyplot as plt
import numpy as np
from dataset import DataLoader
from model import ed_model
import multiprocessing as mp
from sklearn.model_selection import ParameterGrid
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, \
    EarlyStopping, ModelCheckpoint, TerminateOnNaN, TensorBoard
import time

dict_config = {
    "sliding_encoder": [8, 12, 16, 20, 24, 28, 32, 36, 40],
    "sliding_decoder": [1, 2, 3, 4, 5, 6, 7],
    "layer_sizes_ed": [[8], [16], [32], [64], [8, 4], [16, 8], [16, 4], [32, 16], [32, 4], [64, 32], [64, 16]],
    # "layer_sizes_f": [[4], [8], [16], [32], [64], [8, 4], [16, 8], [16, 4], [32, 16], [32, 8], [64, 32], [64, 16],
    #                   [64, 8]],
    "activation": ['tanh', 'sigmoid'],
    "optimizer": ['adam', 'rmsprop'],
    "batch_size": [8, 16, 32, 64, 128],
    "cell_type": ['lstm'],
    "epochs": [500],
    "recurrent_dropout": [0.05],
    "dropout": [0.05],
    "learning_rate": [0.0001, 0.001, 0.01],
    "patience": [15],
}

dataset_name = 'gg_trace'
usecols = [3, 4]  # TODO: fix for each dataset
# data_loader = DataLoader('datasets/wc98/wc98_workload_5min.csv', usecols=usecols, is_scale=True, is_log=True, is_diff=True)
# data_loader = DataLoader('datasets/traffic/internet-traffic-data-in-bits-fr_EU_5m.csv', usecols=usecols, is_scale=True, is_log=True, is_diff=True)
data_loader = DataLoader('datasets/gg_trace/5.csv', usecols=usecols, is_scale=True, is_log=True, is_diff=True)


def run_ed(params):
    train, test = data_loader.get_data(params['sliding_encoder'], params['sliding_decoder'])

    start_time = time.time()
    model_name = "sle({})_sld({})_lsed({})_lsf()_ac({})_opt({})_rd({})_drop({})_bs({})_lr({})_ct({})_pat({})".format(
        params['sliding_encoder'],
        params['sliding_decoder'],
        params['layer_sizes_ed'],
        # params['layer_sizes_f'],
        params['activation'],
        params['optimizer'],
        params['recurrent_dropout'],
        params['dropout'],
        params['batch_size'],
        params['learning_rate'],
        params['cell_type'],
        params['patience']
    )

    print('Runing config:' + model_name)
    model = ed_model(params['layer_sizes_ed'],
                     input_shape_e=(params['sliding_encoder'], len(usecols)),
                     input_shape_d=(params['sliding_decoder'], len(usecols)),
                     activation=params['activation'],
                     dropout=params['dropout'],
                     recurrent_dropout=params['recurrent_dropout'],
                     optimizer=params['optimizer'],
                     learning_rate=params['learning_rate'],
                     multi_output_decoder=True)

    callbacks = [
        # ModelCheckpoint('logs/' + dataset_name + '/best_model.hdf5', save_best_only=True),
        EarlyStopping(patience=params['patience'], monitor='val_loss', restore_best_weights=True),
        # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0),
        TerminateOnNaN()
    ]

    model.fit(train[0], train[1], validation_split=0.2,
              batch_size=params['batch_size'],
              epochs=params['epochs'],
              shuffle=False,
              callbacks=callbacks,
              verbose=0)

    preds = model.predict(test[0])
    preds = preds[:, :1, 0]
    preds_inv = data_loader.inverse_scale(preds)
    preds_inv = data_loader.inverse_log_difference(preds_inv, test[-2])

    # y_test_inv = test[1][:, :1, 0]
    # y_test_inv = data_loader.inverse_scale(y_test_inv)
    # y_test_inv = data_loader.inverse_log_difference(y_test_inv, test[-2])
    y_test_inv = ground_truth = test[-1]
    mae = round(np.mean(np.abs(np.subtract(preds_inv, y_test_inv))), 4)

    delta_time = time.time() - start_time
    with open('logs/' + dataset_name + '/mae.csv', 'a') as f: #TODO: Fix for each dataset
        f.write("{};{:.5f};{:.2f}\n".format(model_name, mae, delta_time))
    plt.plot(y_test_inv, label='actual', color='#fc6b00', linestyle='solid')
    plt.plot(preds_inv, label='predict', color='blue', linestyle='solid')
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend()
    plt.title('mae={:.2f}'.format(mae))
    # plt.show()
    plt.savefig('logs/' + dataset_name + '/' + str(mae) + "_" + model_name + '_predict_ed.png')#TODO: Fix for each dataset
    plt.clf()

    del model



def mutil_running(list_configs, n_jobs=1):
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    pool = mp.Pool(n_jobs)

    num_configs = len(list_configs)
    config_per_map = 64
    n_maps = num_configs // config_per_map
    if num_configs % config_per_map != 0:
        n_maps += 1

    for i in range(n_maps):
        list_configs_map = list_configs[i * config_per_map: (i + 1) * config_per_map]
        pool.map(run_ed, list_configs_map)

    pool.close()
    pool.join()
    pool.terminate()


test_config = {
    'sliding_encoder': 4,
    'sliding_decoder': 1,
    'layer_sizes_ed': [16, 4],
    'layer_sizes_f': [64, 16],
    'activation': 'tanh',
    'optimizer': 'rmsprop',
    'recurrent_dropout': 0.05,
    'dropout': 0.05,
    'batch_size': 4,
    'learning_rate': 0.01,
    'epochs': 2,
    'cell_type': 'lstm',
    'patience': 5
}

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', default=False, type=bool, choices=[True, False])
parser.add_argument('--n_jobs', default=1, type=int)
parser.add_argument('--n_configs', default=1, type=int)
args = parser.parse_args()

list_config = np.random.choice(list(ParameterGrid(dict_config)), size=args.n_configs)

if args.test:
    run_ed(test_config)
else:
    mutil_running(list_configs=list_config, n_jobs=args.n_jobs)