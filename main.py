import pandas as pd
from dataset import *


tf.logging.set_verbosity(tf.logging.INFO)

TRAIN_PATH = 'data/tf_train.csv'
VALIDATION_PATH = 'data/tf_validat1ion.csv'
TEST_PATH = 'data/test_tf.csv'


CSV_COLUMNS = ['key', 'fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
               'dropoff_latitude', 'passenger_count']
LABEL_COLUMN = 'fare_amount'
DEFAULTS = [['nokey'], [1.0], ['date'], [-74.0], [40.0], [-74.0], [40.7], [1.0]]

INPUT_COLUMNS = [
    # raw data columns
    tf.feature_column.numeric_column('pickup_longitude'),
    tf.feature_column.numeric_column('pickup_latitude'),
    tf.feature_column.numeric_column('dropoff_longitude'),
    tf.feature_column.numeric_column('dropoff_latitude'),
    tf.feature_column.numeric_column('passenger_count'),

    # engineered columns
]

feature_cols = add_more_features(INPUT_COLUMNS)

run_config = tf.estimator.RunConfig(model_dir='models/', save_summary_steps=5000, save_checkpoints_steps=5000)
train_spec = tf.estimator.TrainSpec(input_fn=get_train(TRAIN_PATH, CSV_COLUMNS, LABEL_COLUMN, default_value=DEFAULTS),
                                    max_steps=20000)
eval_spec = tf.estimator.EvalSpec(input_fn=get_valid(VALIDATION_PATH, CSV_COLUMNS, LABEL_COLUMN,
                                                     default_value=DEFAULTS), steps=100, throttle_secs=300)

estimator = tf.estimator.DNNRegressor(hidden_units=[128, 64, 32], feature_columns=feature_cols, config=run_config)
tf.estimator.train_and_evaluate(estimator, train_spec=train_spec, eval_spec=eval_spec)


# predictions = model.predict(input_fn=get_test())
test_raw = pd.read_csv('data/test.csv')
test = test_raw.drop(['pickup_datetime', 'key'], axis=1).as_matrix()
prediction = estimator.predict(numpy_test_input_fn(test, ['pickup_longitude', 'pickup_latitude',
                                                       'dropoff_longitude', 'dropoff_latitude', 'passenger_count']))

prediction_df = pd.DataFrame(prediction)

output_submission(test_raw, prediction_df, 'key', 'fare_amount', 'submission.csv')
