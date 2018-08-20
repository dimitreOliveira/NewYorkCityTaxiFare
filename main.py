import pandas as pd
from dataset import *


tf.logging.set_verbosity(tf.logging.INFO)

# Parameters
TRAIN_PATH = 'data/tf_train2.csv'
VALIDATION_PATH = 'data/tf_validation2.csv'
TEST_PATH = 'data/tf_test.csv'
MODEL_DIR = 'models/model8'
SUBMISSION_NAME = 'submission8.csv'


CSV_COLUMNS = ['key', 'fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
               'dropoff_latitude', 'passenger_count', 'year', 'month', 'day', 'hour']
LABEL_COLUMN = 'fare_amount'
DEFAULTS = [['nokey'], [1.0], ['2009-06-15 17:26:21 UTC'], [-74.0], [40.0], [-74.0], [40.7], [1.0], [2009], [6], [15], [17]]
INPUT_COLUMNS = [
    # raw data columns
    tf.feature_column.numeric_column('pickup_longitude'),
    tf.feature_column.numeric_column('pickup_latitude'),
    tf.feature_column.numeric_column('dropoff_longitude'),
    tf.feature_column.numeric_column('dropoff_latitude'),
    tf.feature_column.numeric_column('passenger_count'),

    # engineered columns
    tf.feature_column.numeric_column('year'),
    tf.feature_column.categorical_column_with_identity('month', num_buckets=12),
    tf.feature_column.categorical_column_with_identity('day', num_buckets=31),
    tf.feature_column.categorical_column_with_identity('hour', num_buckets=24),

    # tensorflow engineered columns
    tf.feature_column.numeric_column('latdiff'),
    tf.feature_column.numeric_column('londiff'),
    tf.feature_column.numeric_column('euclidean')
]


estimator = build_estimator(MODEL_DIR, 16, [64, 64, 64, 8], INPUT_COLUMNS)

run_config = tf.estimator.RunConfig(model_dir=MODEL_DIR, save_summary_steps=5000, save_checkpoints_steps=5000)
train_spec = tf.estimator.TrainSpec(input_fn=get_train(TRAIN_PATH, CSV_COLUMNS, LABEL_COLUMN, default_value=DEFAULTS),
                                    max_steps=20000)
eval_spec = tf.estimator.EvalSpec(input_fn=get_valid(VALIDATION_PATH, CSV_COLUMNS, LABEL_COLUMN,
                                                     default_value=DEFAULTS), steps=100, throttle_secs=300)

tf.estimator.train_and_evaluate(estimator, train_spec=train_spec, eval_spec=eval_spec)

# Make predictions
# predictions = model.predict(input_fn=get_test())
test_raw = pd.read_csv('data/test_processed.csv')
add_engineered(test_raw)
test = test_raw.drop(['pickup_datetime', 'key'], axis=1).as_matrix()

# prediction = estimator.predict(numpy_test_input_fn(test, ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
#                                                           'dropoff_latitude', 'passenger_count', 'latdiff', 'londiff',
#                                                           'euclidean']))
prediction = estimator.predict(pandas_test_input_fn(test_raw))

prediction_df = pd.DataFrame(prediction)
output_submission(test_raw, prediction_df, 'key', 'fare_amount', SUBMISSION_NAME)
