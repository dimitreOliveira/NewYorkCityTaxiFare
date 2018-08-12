import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from dataset import load_data, pandas_train_input_fn, pandas_test_input_fn, make_feature_cols, output_submission


tf.logging.set_verbosity(tf.logging.INFO)

TRAIN_PATH = 'data/train2.csv'
TEST_PATH = 'data/test.csv'

train_raw, test = load_data(TRAIN_PATH, TEST_PATH)
labels = train_raw['fare_amount']
train_raw = train_raw.drop(['pickup_datetime', 'key', 'fare_amount'], axis=1)

x_train, x_valid, y_train, y_valid = train_test_split(train_raw, labels, train_size=0.8, random_state=1)


# model = tf.estimator.LinearRegressor(make_feature_cols(train.columns), model_dir='models/')
# model.train(pandas_train_input_fn(train, labels), steps=500)

model = tf.estimator.DNNRegressor(hidden_units=[32, 8, 2], feature_columns=make_feature_cols(x_train.columns),
                                  model_dir='models/')
model.train(input_fn=pandas_train_input_fn(x_train, y_train), steps=1000)

evaluation = model.evaluate(pandas_train_input_fn(x_valid, y_valid))

for key, value in sorted(evaluation.items()):
    print('%s: %s' % (key, value))


predictions = model.predict(pandas_test_input_fn(test))
# for i in range(10):
#   print(predictions.__next__())

predictions_df = pd.DataFrame(predictions)

output_submission(test, predictions_df, 'key', 'fare_amount', 'submission2.csv')

# def print_rmse(model, name, df):
#   metrics = model.evaluate(input_fn = make_input_fn(df, 1))
#   print('RMSE on {} dataset = {}'.format(name, np.sqrt(metrics['average_loss'])))
# print_rmse(model, 'validation', df_valid)

# # scale values
# standard_scaler = preprocessing.StandardScaler()
# train_pre = standard_scaler.fit_transform(train_pre)
# test_pre = standard_scaler.fit_transform(test_pre)
