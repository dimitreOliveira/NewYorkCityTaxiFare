from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from dataset import *


tf.logging.set_verbosity(tf.logging.INFO)

# TRAIN_PATH = 'data/train3.csv'
TRAIN_PATH = 'data/train_tf.csv'
VALIDATION_PATH = 'data/train3.csv'
TEST_PATH = 'data/test_tf.csv'


CSV_COLUMNS = ['key', 'fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
               'dropoff_latitude', 'passenger_count']
LABEL_COLUMN = 'fare_amount'
DEFAULTS = [['nokey'], [1.0], ['date'], [-74.0], [40.0], [-74.0], [40.7], [1.0]]


def read_dataset(filename, mode, features_cols, label_col, batch_size=512):
    def _input_fn():
        def decode_csv(value_column):
            columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
            features = dict(zip(features_cols, columns))
            label = features.pop(label_col)
            return features, label

        # Create list of file names that match "glob" pattern (i.e. data_file_*.csv)
        filenames_dataset = tf.data.Dataset.list_files(filename)
        # Read lines from text files
        textlines_dataset = filenames_dataset.flat_map(tf.data.TextLineDataset)
        # Parse text lines as comma-separated values (CSV)
        dataset = textlines_dataset.map(decode_csv)

        # Note:
        # use tf.data.Dataset.flat_map to apply one to many transformations (here: filename -> text lines)
        # use tf.data.Dataset.map      to apply one to one  transformations (here: text line -> feature list)

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


def get_train():
    return read_dataset(TRAIN_PATH, mode=tf.estimator.ModeKeys.TRAIN, features_cols=CSV_COLUMNS, label_col=LABEL_COLUMN)


def get_valid():
    return read_dataset(VALIDATION_PATH, mode=tf.estimator.ModeKeys.EVAL, features_cols=CSV_COLUMNS, label_col=LABEL_COLUMN)


def get_test():
    return read_dataset(TEST_PATH, mode=tf.estimator.ModeKeys.PREDICT, features_cols=CSV_COLUMNS, label_col=LABEL_COLUMN)


INPUT_COLUMNS = [
    tf.feature_column.numeric_column('pickup_longitude'),
    tf.feature_column.numeric_column('pickup_latitude'),
    tf.feature_column.numeric_column('dropoff_longitude'),
    tf.feature_column.numeric_column('dropoff_latitude'),
    tf.feature_column.numeric_column('passenger_count'),
]


def add_more_features(feats):
    # Nothing to add (yet!)
    return feats


feature_cols = add_more_features(INPUT_COLUMNS)

run_config = tf.estimator.RunConfig(
    model_dir='models/',
    save_summary_steps=5000,
    save_checkpoints_steps=5000
)
train_spec = tf.estimator.TrainSpec(input_fn=get_train(),
                                    max_steps=20000)
eval_spec = tf.estimator.EvalSpec(input_fn=get_valid(),
                                  steps=100,
                                  throttle_secs=300)

# estimator = tf.estimator.LinearRegressor(feature_columns=feature_cols, config=run_config)
estimator = tf.estimator.DNNRegressor(hidden_units=[128, 64, 32], feature_columns=feature_cols, config=run_config)
tf.estimator.train_and_evaluate(estimator, train_spec=train_spec, eval_spec=eval_spec)


# predictions = model.predict(input_fn=get_test())
test_raw = pd.read_csv('data/test.csv')
test = test_raw.drop(['pickup_datetime', 'key'], axis=1).as_matrix()
prediction = estimator.predict(numpy_test_input_fn(test, ['pickup_longitude', 'pickup_latitude',
                                                       'dropoff_longitude', 'dropoff_latitude', 'passenger_count']))

prediction_df = pd.DataFrame(prediction)

output_submission(test_raw, prediction_df, 'key', 'fare_amount', 'submission6.csv')

# print('loading data')
# train_raw, test_raw = load_data(TRAIN_PATH, TEST_PATH)
#
# # pre process
# train_raw = train_raw[train_raw['fare_amount'] > 0]
# train_raw = train_raw.dropna()
#
# # get labels
# labels = train_raw['fare_amount']
#
# # drop unwanted columns
# train_raw = train_raw.drop(['pickup_datetime', 'key', 'fare_amount'], axis=1)
# test = test_raw.drop(['pickup_datetime', 'key'], axis=1)
#
# features = make_feature_cols(train_raw.columns)
#
# # split data
# x_train, x_valid, y_train, y_valid = train_test_split(train_raw.values, labels.values, train_size=0.9, random_state=1)
#
# # scale values
# standard_scaler = preprocessing.StandardScaler()
# x_train = standard_scaler.fit_transform(x_train)
# x_valid = standard_scaler.transform(x_valid)
# x_test = standard_scaler.transform(test)
