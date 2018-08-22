import numpy as np
import tensorflow as tf


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


def get_train(data_path, features, label, default_value, batch_size=512):
    return read_dataset(data_path, mode=tf.estimator.ModeKeys.TRAIN, features_cols=features, label_col=label,
                        default_value=default_value, batch_size=batch_size)


def get_valid(data_path, features, label, default_value, batch_size=512):
    return read_dataset(data_path, mode=tf.estimator.ModeKeys.EVAL, features_cols=features, label_col=label,
                        default_value=default_value, batch_size=batch_size)


def get_test(data_path, features, label, default_value, batch_size=512):
    return read_dataset(data_path, mode=tf.estimator.ModeKeys.PREDICT, features_cols=features, label_col=label,
                        default_value=default_value, batch_size=batch_size)


def add_engineered(features):
    # Feature engineering as data is feed
    lat1 = features['pickup_latitude']
    lat2 = features['dropoff_latitude']
    lon1 = features['pickup_longitude']
    lon2 = features['dropoff_longitude']
    latdiff = (lat1 - lat2)
    londiff = (lon1 - lon2)
    euclidean = (latdiff ** 2 + londiff ** 2) ** 0.5

    # Add new features
    features['latdiff'] = latdiff
    features['londiff'] = londiff
    features['euclidean'] = euclidean

    return features


def build_estimator(model_dir, nbuckets, hidden_units, input_columns):
    # Input columns
    (plon, plat, dlon, dlat, pcount, year, month, day, hour, latdiff, londiff, euclidean) = input_columns

    # Bucketize the lats & lons
    latbuckets = np.linspace(38.0, 42.0, nbuckets).tolist()
    lonbuckets = np.linspace(-76.0, -72.0, nbuckets).tolist()
    b_plat = tf.feature_column.bucketized_column(plat, latbuckets)
    b_dlat = tf.feature_column.bucketized_column(dlat, latbuckets)
    b_plon = tf.feature_column.bucketized_column(plon, lonbuckets)
    b_dlon = tf.feature_column.bucketized_column(dlon, lonbuckets)

    # Feature cross
    ploc = tf.feature_column.crossed_column([b_plat, b_plon], nbuckets * nbuckets)
    dloc = tf.feature_column.crossed_column([b_dlat, b_dlon], nbuckets * nbuckets)
    pd_pair = tf.feature_column.crossed_column([ploc, dloc], nbuckets ** 4)

    # Wide columns and deep columns
    wide_columns = [
        # Feature crosses
        dloc, ploc, pd_pair,

        # Sparse columns
        month, day, hour,

        # Anything with a linear relationship
        year, pcount
    ]

    deep_columns = [
        # Embedding_column to "group" together
        tf.feature_column.embedding_column(pd_pair, 10),

        # Numeric columns
        plat, plon, dlat, dlon,
        latdiff, londiff, euclidean
    ]

    estimator = tf.estimator.DNNLinearCombinedRegressor(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units)

    # add extra evaluation metric for hyperparameter tuning
    estimator = tf.contrib.estimator.add_metrics(estimator, add_eval_metrics)

    return estimator


def add_eval_metrics(labels, predictions):
    pred_values = predictions['predictions']
    return {
        'rmse': tf.metrics.root_mean_squared_error(labels, pred_values),
        'mae': tf.metrics.mean_absolute_error(labels, pred_values)
    }


def pandas_train_input_fn(df, label):
    return tf.estimator.inputs.pandas_input_fn(
        x=df,
        y=label,
        batch_size=128,
        num_epochs=100,
        shuffle=True,
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


def pandas_test_input_fn(df):
    return tf.estimator.inputs.pandas_input_fn(
        x=df,
        y=None,
        batch_size=128,
        num_epochs=1,
        shuffle=True,
        queue_capacity=1000
    )


def numpy_test_input_fn(df, features):
    return tf.estimator.inputs.numpy_input_fn(
        x={features[i]: df[:, i] for i in range(len(features))},
        y=None,
        batch_size=128,
        num_epochs=1,
        shuffle=True,
        queue_capacity=1000
    )


def make_feature_cols(features):
    input_columns = [tf.feature_column.numeric_column(f) for f in features]
    return input_columns


def output_submission(df, prediction_df, id_column, prediction_column, file_name):
    df[prediction_column] = prediction_df['predictions'].apply(lambda x: x[0])
    df[[id_column, prediction_column]].to_csv(('submissions/%s' % file_name), index=False)
    print('Output complete')
