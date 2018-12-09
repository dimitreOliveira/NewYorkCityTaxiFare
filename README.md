![](https://cdn.static-economist.com/sites/default/files/images/2015/08/blogs/graphic-detail/20150815_blp501.jpg)

# Deep Learning regression with Tensorflow

## About the repository
The goal here is use the Tensorflow API and create a end-to-end project, from data loading to model predictions, and use the Kaggle "New York City Taxi Fare Prediction competition" as the data source.

### What you will find
* Data preprocessing and cleaning using the "csv" library.
* Data batch loading using Tensorflow.
* Preprocess each batch by time with Tensorflow.
* Deep learning models using the estimator API from Tensorflow.
* Tensorflow estimator API wrapping Keras models.
* Model validation and analysis using Tensorboard.
* Model prediction by batch with Tensorflow.

### New York City Taxi Fare Prediction
#### Can you predict a rider's taxi fare?

Kaggle competition: https://www.kaggle.com/c/new-york-city-taxi-fare-prediction

### Overview
In this playground competition, hosted in partnership with Google Cloud and Coursera, you are tasked with predicting the fare amount (inclusive of tolls) for a taxi ride in New York City given the pickup and dropoff locations. While you can get a basic estimate based on just the distance between the two points, this will result in an RMSE of $5-$8, depending on the model used (see the starter code for an example of this approach in Kernels). Your challenge is to do better than this using Machine Learning techniques!

To learn how to handle large datasets with ease and solve this problem using TensorFlow, consider taking the Machine Learning with TensorFlow on Google Cloud Platform specialization on Coursera -- the taxi fare problem is one of several real-world problems that are used as case studies in the series of courses.

### Dependencies:
* [csv](https://docs.python.org/3/library/csv.html)
* [keras](https://keras.io/)
* [numpy](http://www.numpy.org/)
* [pandas](http://pandas.pydata.org/)
* [sklearn](https://scikit-learn.org/stable/)
* [datetime](https://docs.python.org/2/library/datetime.html)
* [tensorflow](https://www.tensorflow.org/)
* [matplotlib](http://matplotlib.org/)

### To-Do:
* Fix prediction by batch "e.g.: estimator.predict() using input_fn=read_dataset() as parameter"
* Efficiently use Keras models wrapped by the Tensorflow estimator API.
