import pandas as pd
import tensorflow as tf


def load_data(train_path, test_path):
    """
    method for data loading
    :param train_path: path for the train set file
    :param test_path: path for the test set file
    :return: a 'pandas' array for each set
    """

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    print("number of training examples = " + str(train_data.shape[0]))
    print("number of test examples = " + str(test_data.shape[0]))
    print("train shape: " + str(train_data.shape))
    print("test shape: " + str(test_data.shape))

    return train_data, test_data


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
