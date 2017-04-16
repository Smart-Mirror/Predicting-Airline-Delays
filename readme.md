# Predicting Airline Delays

Using Hadoop predicting the airline delays from O'Hare airport with data from 2007 and 2008. Using Pig scripts, built a feature matrix with which we train and predict the airline delay with an accuracy of about 80%


#### Project Details

- Built a model for predicting the airline delays with an accuracy of ~80%
- Used the airline dataset with 7.4M flight records from the UCI Repo
- Utilized Pydoop for implementing MapReduce to build the feature matrix
- Used Pig scripts to generate the features
- Built using Python, Scikit-Learn, Pig, Hadoop, HDFS, AWS EMR, IPython

#### Specifications

- Python 2.7
- Hadoop 2.7.3
- Scikit-Learn
- Pandas
- Linear Regression
- Random Forests

#### Algorithm

1. Exploring the raw data to determine various properties of features and how predictive these features might be for the task at hand.

2. Using PIG and Python to prepare the feature matrix from the raw data. We perform 3 iterations. With each iteration, we improve our feature set, resulting in better overall predictive performance. For example, in the 3rd iteration, we enrich the input data with weather information, resulting in predictive features such as temperature, snow conditions, or wind speed.

3. Using Python’s Scikit-learn, we build various models, such as Logistic Regression or Random Forest.

4. Using Scikit-learn, we evaluate performance of the models and compare between iterations.


#### Dataset

- [Airline dataset](http://stat-computing.org/dataexpo/2009/the-data.html)


