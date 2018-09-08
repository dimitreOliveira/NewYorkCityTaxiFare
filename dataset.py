import tensorflow as tf
import numpy as np

def read_dataset(filename, mode, features_cols, label_col, default_value, batch_size=512):
    def _input_fn():
        def decode_csv(value_column):
            columns = tf.decode_csv(value_column, record_defaults=default_value)
            features = dict(zip(features_cols, columns))
            label = features.pop(label_col)
            return add_engineered(features), label

        # Create list of file names that match "glob" pattern (i.e. data_file_*.csv)
        filenames_dataset = tf.data.Dataset.list_files(filename)
        # Read lines from text files
        # use tf.data.Dataset.flat_map to apply one to many transformations (here: filename -> text lines)
        textlines_dataset = filenames_dataset.flat_map(tf.data.TextLineDataset)
        # Parse text lines as comma-separated values (CSV)
        # use tf.data.Dataset.map      to apply one to one  transformations (here: text line -> feature list)
        dataset = textlines_dataset.map(decode_csv)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None  # loop indefinitely
            dataset = dataset.shuffle(buffer_size=10 * batch_size)
        elif mode == tf.estimator.ModeKeys.EVAL:
            num_epochs = 1  # end-of-input after this
        else:
            num_epochs = 1  # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        batch_features, batch_labels = dataset.make_one_shot_iterator().get_next()

        return batch_features, batch_labels

    return _input_fn


def add_engineered(features):
    # Feature engineering as data is fed
    lat1 = features['pickup_latitude']
    lat2 = features['dropoff_latitude']
    lon1 = features['pickup_longitude']
    lon2 = features['dropoff_longitude']
    latdiff = (lat1 - lat2)
    londiff = (lon1 - lon2)
    euclidean = (latdiff ** 2 + londiff ** 2) ** 0.5
    manhattan = np.abs(lat2 - lat1) + np.abs(lon2 - lon1)

    # Add new features
    features['latdiff'] = latdiff
    features['londiff'] = londiff
    features['euclidean'] = euclidean
    features['manhattan'] = manhattan

    return features


def pandas_train_input_fn(df, label):
    return tf.estimator.inputs.pandas_input_fn(
        x=df,
        y=label,
        batch_size=128,
        num_epochs=100,
        shuffle=True,
        queue_capacity=1000
    )


def pandas_test_input_fn(df):
    return tf.estimator.inputs.pandas_input_fn(
        x=df,
        y=None,
        batch_size=128,
        num_epochs=1,
        shuffle=False,
        queue_capacity=1000
    )


def numpy_train_input_fn(df, features, label):
    return tf.estimator.inputs.numpy_input_fn(
        x={features[i]: df[:, i] for i in range(len(features))},
        y=label,
        batch_size=128,
        num_epochs=100,
        shuffle=True,
        queue_capacity=1000
    )


def numpy_test_input_fn(df, features):
    return tf.estimator.inputs.numpy_input_fn(
        x={features[i]: df[:, i] for i in range(len(features))},
        y=None,
        batch_size=128,
        num_epochs=1,
        shuffle=False,
        queue_capacity=1000
    )


def output_submission(df, prediction_df, id_column, prediction_column, file_name):
    df[prediction_column] = prediction_df['predictions'].apply(lambda x: x[0])
    df[[id_column, prediction_column]].to_csv(('submissions/%s' % file_name), index=False)
    print('Output complete')


# def read_dataset(filename, mode, features_cols, label_col, default_value, batch_size=512):
def keras_read_dataset(filename, mode, TIMESERIES_COL, DEFAULTS, label_index=1, batch_size=512):
    def _input_fn():
        # Provide the ability to decode a CSV
        def decode_csv(line):
            # all_data is a list of scalar tensors
            all_data = tf.decode_csv(line, record_defaults=DEFAULTS)
            inputs = all_data
            labels = inputs.pop(label_index)  # labels are the column of index 1
            print('-----------1')
            print(inputs)

            # # Convert each list of rank R tensors to one rank R+1 tensor
            # inputs = tf.stack(inputs, axis=0)
            # labels = tf.stack(labels, axis=0)

            # Convert input R+1 tensor into a feature dictionary of one R+1 tensor
            # features = {TIMESERIES_COL: inputs}
            # features = inputs

            return {TIMESERIES_COL: inputs}, labels

        # Create list of files that match pattern
        file_list = tf.gfile.Glob(filename)

        # Create dataset from file list
        dataset = tf.data.TextLineDataset(file_list).map(decode_csv)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None  # indefinitely
            dataset = dataset.shuffle(buffer_size=10 * batch_size)
        else:
            num_epochs = 1  # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = iterator.get_next()
        return batch_features, batch_labels
    return _input_fn
