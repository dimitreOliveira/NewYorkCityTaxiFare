import pandas as pd
from model import *
from dataset import *


tf.logging.set_verbosity(tf.logging.INFO)

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
DEFAULTS = [['nokey'], [1.0], ['2009-06-15 17:26:21 UTC'], [-74.0], [40.0], [-74.0], [40.7], [1.0], [2009], [6], [15], [17], [1]]
INPUT_COLUMNS = [
    # raw data columns
    tf.feature_column.numeric_column('pickup_longitude'),
    tf.feature_column.numeric_column('pickup_latitude'),
    tf.feature_column.numeric_column('dropoff_longitude'),
    tf.feature_column.numeric_column('dropoff_latitude'),
    tf.feature_column.numeric_column('passenger_count'),

    # engineered columns'1
    tf.feature_column.numeric_column('year'),
    tf.feature_column.categorical_column_with_identity('month', num_buckets=13),
    tf.feature_column.categorical_column_with_identity('day', num_buckets=32),
    tf.feature_column.categorical_column_with_identity('hour', num_buckets=24),
    tf.feature_column.categorical_column_with_identity('weekday', num_buckets=7),

    # tensorflow engineered columns
    tf.feature_column.numeric_column('latdiff'),
    tf.feature_column.numeric_column('londiff'),
    tf.feature_column.numeric_column('euclidean')
]


optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
estimator = build_estimator(MODEL_DIR, 16, [100, 50, 20], optimizer, INPUT_COLUMNS)

run_config = tf.estimator.RunConfig(model_dir=MODEL_DIR, save_summary_steps=5000, save_checkpoints_steps=5000)
train_spec = tf.estimator.TrainSpec(input_fn=read_dataset(TRAIN_PATH, mode=tf.estimator.ModeKeys.TRAIN,
                                                          features_cols=CSV_COLUMNS, label_col=LABEL_COLUMN,
                                                          default_value=DEFAULTS, batch_size=BATCH_SIZE),
                                    max_steps=100000)
eval_spec = tf.estimator.EvalSpec(input_fn=read_dataset(TRAIN_PATH, mode=tf.estimator.ModeKeys.EVAL,
                                                        features_cols=CSV_COLUMNS, label_col=LABEL_COLUMN,
                                                        default_value=DEFAULTS, batch_size=BATCH_SIZE),
                                  steps=1000, throttle_secs=300)

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
