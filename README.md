![](https://cdn.static-economist.com/sites/default/files/images/2015/08/blogs/graphic-detail/20150815_blp501.jpg)

# Deep Learning regression with Tensorflow

## About the repository
The goal here is use the Tensorflow API and create a end-to-end project, from data loading to model predictions, and use the Kaggle "New York City Taxi Fare Prediction competition" as the data source.

### What you will find
* Data preprocessing and cleaning using the "csv" library. [[link]](https://github.com/dimitreOliveira/NewYorkCityTaxiFare/blob/master/data_querys.py)
* Data batch loading using Tensorflow. [[link]](https://github.com/dimitreOliveira/NewYorkCityTaxiFare/blob/master/dataset.py)
* Preprocess each batch by time with Tensorflow. [[link]](https://github.com/dimitreOliveira/NewYorkCityTaxiFare/blob/master/dataset.py)
* Deep learning models using the estimator API from Tensorflow. [[link]](https://github.com/dimitreOliveira/NewYorkCityTaxiFare/blob/master/model.py)
* Tensorflow estimator API wrapping Keras models. [[link]](https://github.com/dimitreOliveira/NewYorkCityTaxiFare/blob/master/keras_model.py)
* Model validation and analysis using Tensorboard. [[link]](https://github.com/dimitreOliveira/NewYorkCityTaxiFare/blob/master/tensorflow_model.py)
* Model prediction by batch with Tensorflow. [[link]](https://github.com/dimitreOliveira/NewYorkCityTaxiFare/blob/master/tensorflow_model.py)

### New York City Taxi Fare Prediction
#### Can you predict a rider's taxi fare?

Kaggle competition: https://www.kaggle.com/c/new-york-city-taxi-fare-prediction

### Overview
In this playground competition, hosted in partnership with Google Cloud and Coursera, you are tasked with predicting the fare amount (inclusive of tolls) for a taxi ride in New York City given the pickup and dropoff locations. While you can get a basic estimate based on just the distance between the two points, this will result in an RMSE of $5-$8, depending on the model used (see the starter code for an example of this approach in Kernels). Your challenge is to do better than this using Machine Learning techniques!

To learn how to handle large datasets with ease and solve this problem using TensorFlow, consider taking the Machine Learning with TensorFlow on Google Cloud Platform specialization on Coursera -- the taxi fare problem is one of several real-world problems that are used as case studies in the series of courses.

### Dependencies:
* [Keras](https://keras.io/)
* [NumPy](http://www.numpy.org/)
* [Pandas](http://pandas.pydata.org/)
* [SkLearn](https://scikit-learn.org/stable/)
* [Tensorflow](https://www.tensorflow.org/)
* [Matplotlib](http://matplotlib.org/)

### To-Do:
* Fix prediction by batch "e.g.: estimator.predict() using input_fn=read_dataset() as parameter"
* Efficiently use Keras models wrapped by the Tensorflow estimator API.
