import pandas as pd
from model import *
from dataset import *


tf.logging.set_verbosity(tf.logging.INFO)

# Parameters
TRAIN_PATH = 'data/tf_train.csv'
VALIDATION_PATH = 'data/tf_validation.csv'
TEST_PATH = 'data/test_processed.csv'
MODEL_DIR = 'models/model22'
SUBMISSION_NAME = 'submission22.csv'


LEARNING_RATE = 0.0001
HIDDEN_UNITS = [256, 128, 64, 32, 16, 8]
STEPS = 300000
BATCH_SIZE = 512
CSV_COLUMNS = ['key', 'fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
               'dropoff_latitude', 'passenger_count', 'year', 'month', 'day', 'hour', 'weekday', 'night', 'late_night']
LABEL_COLUMN = 'fare_amount'
DEFAULTS = [['nokey'], [1.0], ['2009-06-15 17:26:21 UTC'], [-74.0], [40.0], [-74.0], [40.7], [1.0], [2009], [6], [15],
            [17], [1], [1], [1]]
INPUT_COLUMNS = [
    # raw data columns
    tf.feature_column.numeric_column('pickup_longitude'),
    tf.feature_column.numeric_column('pickup_latitude'),
    tf.feature_column.numeric_column('dropoff_longitude'),
    tf.feature_column.numeric_column('dropoff_latitude'),
    tf.feature_column.numeric_column('passenger_count'),

    # csv engineered columns
    tf.feature_column.numeric_column('year'),
    tf.feature_column.categorical_column_with_identity('month', num_buckets=13),
    tf.feature_column.categorical_column_with_identity('day', num_buckets=32),
    tf.feature_column.categorical_column_with_identity('hour', num_buckets=24),
    tf.feature_column.categorical_column_with_identity('weekday', num_buckets=7),
    tf.feature_column.categorical_column_with_identity('night', num_buckets=2),
    tf.feature_column.categorical_column_with_identity('late_night', num_buckets=2),

    # tensorflow engineered columns
    tf.feature_column.numeric_column('latdiff'),
    tf.feature_column.numeric_column('londiff'),
    tf.feature_column.numeric_column('euclidean'),
    tf.feature_column.numeric_column('manhattan')
]


train_spec = tf.estimator.TrainSpec(input_fn=read_dataset(TRAIN_PATH, mode=tf.estimator.ModeKeys.TRAIN,
                                                          features_cols=CSV_COLUMNS, label_col=LABEL_COLUMN,
                                                          default_value=DEFAULTS, batch_size=BATCH_SIZE),
                                    max_steps=STEPS)
eval_spec = tf.estimator.EvalSpec(input_fn=read_dataset(VALIDATION_PATH, mode=tf.estimator.ModeKeys.EVAL,
                                                        features_cols=CSV_COLUMNS, label_col=LABEL_COLUMN,
                                                        default_value=DEFAULTS, batch_size=BATCH_SIZE),
                                  steps=1000, throttle_secs=300)


optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
# estimator = build_deep_estimator(MODEL_DIR, 16, HIDDEN_UNITS, optimizer, INPUT_COLUMNS)
estimator = build_combined_estimator(MODEL_DIR, 16, HIDDEN_UNITS, optimizer, INPUT_COLUMNS)

tf.estimator.train_and_evaluate(estimator, train_spec=train_spec, eval_spec=eval_spec)

# Make predictions
# predictions = estimator.predict(input_fn=read_dataset(TEST_PATH, mode=tf.estimator.ModeKeys.PREDICT,
#                                                       features_cols=CSV_COLUMNS, label_col=LABEL_COLUMN,
#                                                       default_value=DEFAULTS, batch_size=BATCH_SIZE))
test_raw = pd.read_csv(TEST_PATH)
add_engineered(test_raw)
prediction = estimator.predict(pandas_test_input_fn(test_raw))

prediction_df = pd.DataFrame(prediction)
output_submission(test_raw, prediction_df, 'key', 'fare_amount', SUBMISSION_NAME)
