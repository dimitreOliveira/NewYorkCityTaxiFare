import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras import optimizers
import matplotlib.pyplot as plt
from dataset import *


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column]
    df[[id_column, prediction_column]].to_csv((file_name), index=False)
    print('Output complete')


def plot_loss_accuracy(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


def train_generator(path):
    while True:
        with open(path) as f:
            for line in f:
                # create numpy arrays of input data
                # and labels, from each line in the file
                (key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude,
                 dropoff_latitude, passenger_count, year, month, day, hour, weekday) = line.split(',')

                numeric = (fare_amount, pickup_longitude, pickup_latitude, dropoff_longitude,
                           dropoff_latitude, passenger_count, year, month, day, hour, weekday)

                (fare_amount, pickup_longitude, pickup_latitude, dropoff_longitude,
                 dropoff_latitude, passenger_count, year, month, day, hour, weekday) = [float(x) for x in numeric]

                latdiff = (pickup_latitude - dropoff_latitude)
                londiff = (pickup_longitude - dropoff_longitude)
                euclidean = (latdiff ** 2 + londiff ** 2) ** 0.5

                ploc = pickup_latitude * pickup_longitude
                dloc = dropoff_latitude * dropoff_longitude
                pd_pair = ploc * dloc
                day_hr = weekday * hour

                x = {'pickup_longitude': pickup_longitude, 'pickup_latitude': pickup_latitude,
                     'dropoff_longitude': dropoff_longitude, 'dropoff_latitude': dropoff_latitude,
                     'passenger_count': passenger_count, 'year': year, 'month': month, 'day': day, 'hour': hour,
                     'weekday': weekday, 'latdiff': latdiff, 'londiff': londiff, 'euclidean': euclidean, 'ploc': ploc,
                     'dloc': dloc, 'pd_pair': pd_pair, 'day_hr': day_hr}
                y = {'fare_amount': fare_amount}

                # x = [pickup_longitude, pickup_latitude, dropoff_longitude,  dropoff_latitude,
                #                 passenger_count,  year, month,  day,  hour,  weekday,  latdiff,  londiff,  euclidean,
                #                 ploc, dloc,  pd_pair,  day_hr]
                # y = [fare_amount]

                print('********')
                print(x)
                print(y)
                # print(x.shape)
                # print(y.shape)

                # yield (x[0], y[0])
                yield (x, y)
                # yield x, y


def generator(path, batch_size=32):
    while True:  # Loop forever so the generator never terminates
        with open(path) as f:
            inputs = []
            labels = []
            # for batch_sample in range(batch_size):
            line = f.__next__()
            (key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude,
             dropoff_latitude, passenger_count, year, month, day, hour, weekday) = line.split(',')

            numeric = (fare_amount, pickup_longitude, pickup_latitude, dropoff_longitude,
                       dropoff_latitude, passenger_count, year, month, day, hour, weekday)

            (fare_amount, pickup_longitude, pickup_latitude, dropoff_longitude,
             dropoff_latitude, passenger_count, year, month, day, hour, weekday) = [float(x) for x in numeric]

            latdiff = (pickup_latitude - dropoff_latitude)
            londiff = (pickup_longitude - dropoff_longitude)
            euclidean = (latdiff ** 2 + londiff ** 2) ** 0.5

            ploc = pickup_latitude * pickup_longitude
            dloc = dropoff_latitude * dropoff_longitude
            pd_pair = ploc * dloc
            day_hr = weekday * hour

            x = [pickup_longitude, pickup_latitude, dropoff_longitude,  dropoff_latitude,
                            passenger_count,  year, month,  day,  hour,  weekday,  latdiff,  londiff,  euclidean,
                            ploc, dloc,  pd_pair,  day_hr]
            y = [fare_amount]

            inputs.append(x)
            labels.append(y)

            x_train = np.array(inputs)
            y_train = np.array(labels)

            print(y_train[0])
            yield x_train, y_train


# Parameters
TRAIN_PATH = 'data/tf_train.csv'
VALIDATION_PATH = 'data/tf_validation.csv'
TEST_PATH = 'data/test_processed.csv'
MODEL_DIR = 'models/model11'
SUBMISSION_NAME = 'submission11.csv'


BATCH_SIZE = 512
CSV_COLUMNS = ['key', 'fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
               'dropoff_latitude', 'passenger_count', 'year', 'month', 'day', 'hour', 'weekday']
LABEL_COLUMN = 'fare_amount'
DEFAULTS = [['nokey'], [1.0], ['2009-06-15 17:26:21 UTC'], [-74.0], [40.0], [-74.0], [40.7], [1.0], [2009], [6], [15],
            [17], [1]]
# INPUT_COLUMNS = [
#     # raw data columns
#     tf.feature_column.numeric_column('pickup_longitude'),
#     tf.feature_column.numeric_column('pickup_latitude'),
#     tf.feature_column.numeric_column('dropoff_longitude'),
#     tf.feature_column.numeric_column('dropoff_latitude'),
#     tf.feature_column.numeric_column('passenger_count'),
#
#     # engineered columns'1
#     tf.feature_column.numeric_column('year'),
#     tf.feature_column.categorical_column_with_identity('month', num_buckets=13),
#     tf.feature_column.categorical_column_with_identity('day', num_buckets=32),
#     tf.feature_column.categorical_column_with_identity('hour', num_buckets=24),
#     tf.feature_column.categorical_column_with_identity('weekday', num_buckets=7),
#
#     # tensorflow engineered columns
#     tf.feature_column.numeric_column('latdiff'),
#     tf.feature_column.numeric_column('londiff'),
#     tf.feature_column.numeric_column('euclidean')
# ]


# optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
#
# train_spec = tf.estimator.TrainSpec(input_fn=read_dataset(TRAIN_PATH, mode=tf.estimator.ModeKeys.TRAIN,
#                                                           features_cols=CSV_COLUMNS, label_col=LABEL_COLUMN,
#                                                           default_value=DEFAULTS, batch_size=BATCH_SIZE),
#                                     max_steps=100000)
# eval_spec = tf.estimator.EvalSpec(input_fn=read_dataset(TRAIN_PATH, mode=tf.estimator.ModeKeys.EVAL,
#                                                         features_cols=CSV_COLUMNS, label_col=LABEL_COLUMN,
#                                                         default_value=DEFAULTS, batch_size=BATCH_SIZE),
#                                   steps=1000, throttle_secs=300)

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=17))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

adam = optimizers.adam(lr=0.0001)
model.compile(loss='mse', optimizer=adam, metrics=['mae'])

# history = model.fit_generator(read_dataset(TRAIN_PATH, mode=tf.estimator.ModeKeys.TRAIN, features_cols=CSV_COLUMNS,
#                                            label_col=LABEL_COLUMN, default_value=DEFAULTS, batch_size=BATCH_SIZE),
#                               workers=4, verbose=2, steps_per_epoch=10000)

history = model.fit_generator(generator(TRAIN_PATH), steps_per_epoch=100, epochs=10, shuffle=True)

test_raw = pd.read_csv(TEST_PATH)
add_engineered(test_raw)
# Make prediction
prediction = model.predict(test_raw, batch_size=128, verbose=1)

# output prediction
output_submission(test_raw, prediction, 'key', 'fare_amount', SUBMISSION_NAME)
