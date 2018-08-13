import tensorflow as tf


def read_dataset(filename, mode, features_cols, label_col, default_value, batch_size=512):
    def _input_fn():
        def decode_csv(value_column):
            columns = tf.decode_csv(value_column, record_defaults=default_value)
            features = dict(zip(features_cols, columns))
            label = features.pop(label_col)
            return features, label

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

        return dataset.make_one_shot_iterator().get_next()

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


def add_more_features(feats):
    # Nothing to add (yet!)
    return feats


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
    print('Outputting submission...')
    df[prediction_column] = prediction_df['predictions'].apply(lambda x: x[0])
    df[[id_column, prediction_column]].to_csv(('submissions/%s' % file_name), index=False)
    print('Output complete')
