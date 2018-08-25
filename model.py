import numpy as np
import tensorflow as tf


def build_estimator(model_dir, nbuckets, hidden_units, optimizer, input_columns):
    # Input columns
    (plon, plat, dlon, dlat, pcount, year, month, day, hour, weekday, latdiff, londiff, euclidean) = input_columns

    # Bucketize the lats & lons
    latbuckets = np.linspace(38.0, 42.0, nbuckets).tolist()
    lonbuckets = np.linspace(-76.0, -72.0, nbuckets).tolist()
    b_plat = tf.feature_column.bucketized_column(plat, latbuckets)
    b_dlat = tf.feature_column.bucketized_column(dlat, latbuckets)
    b_plon = tf.feature_column.bucketized_column(plon, lonbuckets)
    b_dlon = tf.feature_column.bucketized_column(dlon, lonbuckets)

    # Feature cross
    ploc = tf.feature_column.crossed_column([b_plat, b_plon], nbuckets ** 2)
    dloc = tf.feature_column.crossed_column([b_dlat, b_dlon], nbuckets ** 2)
    pd_pair = tf.feature_column.crossed_column([ploc, dloc], nbuckets ** 4)
    day_hr = tf.feature_column.crossed_column([weekday, hour], 24 * 7)

    # Wide columns and deep columns
    wide_columns = [
        # Feature crosses
        dloc, ploc, pd_pair, day_hr,

        # Sparse columns
        month, day, hour, weekday,

        # Anything with a linear relationship
        year, pcount
    ]

    deep_columns = [
        # Embedding_column to "group" together
        tf.feature_column.embedding_column(pd_pair, 10),
        tf.feature_column.embedding_column(day_hr, 10),

        # Numeric columns
        plat, plon, dlat, dlon,
        latdiff, londiff, euclidean
    ]

    estimator = tf.estimator.DNNLinearCombinedRegressor(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        dnn_optimizer=optimizer)

    # add extra evaluation metric for hyperparameter tuning
    estimator = tf.contrib.estimator.add_metrics(estimator, add_eval_metrics)

    return estimator


def add_eval_metrics(labels, predictions):
    pred_values = predictions['predictions']
    return {
        'rmse': tf.metrics.root_mean_squared_error(labels, pred_values),
        'mae': tf.metrics.mean_absolute_error(labels, pred_values)
    }
