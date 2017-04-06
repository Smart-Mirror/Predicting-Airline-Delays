import readfile
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

cols = ['year', 'month', 'day', 'dow', 'DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime', 'Carrier', 'FlightNum',
        'TailNum', 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 'ArrDelay', 'DepDelay', 'Origin', 'Dest',
        'Distance', 'TaxiIn', 'TaxiOut', 'Cancelled', 'CancellationCode', 'Diverted', 'CarrierDelay',
        'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']

# Reading from the CSV file using Hadoop HDFS
flights_2007 = readfile.read_csv_from_hdfs('airline/delay/2007.csv', cols)

# Flight data from 2007
print(flights_2007.shape)

# Total flight and delays at the O'hare airport in 2007
df = flights_2007[flights_2007['Origin'] == 'ORD'].dropna(subset=['DepDelay'])
df['DepDelayed'] = df['DepDelay'].apply(lambda x: x >= 15)
print("Total flights: " + str(df.shape[0]))
print("Total delays: " + str(df['DepDelayed'].sum()))

# Average number of delayed flights per month
Month_delays = df[['DepDelayed', 'month']].groupby('month').mean()

# plotting average delays by month
Month_delays.plot(kind='bar')

# Average number of delayed flights by hour
df['hour'] = df['CRSDepTime'].map(lambda x: int(str(int(x)).zfill(4)[:2]))
hour_delays = df[['DepDelayed', 'hour']].groupby('hour').mean()

# Plotting average delays by hour of day
hour_delays.plot(kind='bar')

# Average number of delayed flights per carrier
grouped1 = df[['DepDelayed', 'Carrier']].groupby('Carrier').filter(lambda x: len(x) > 10)
grouped2 = grouped1.groupby('Carrier').mean()
carrier_delays = grouped2.sort(['DepDelayed'], ascending=False)

# Plotting top 15 destination carriers by delay
carrier_delays[:15].plot(kind='bar')

# read files
cols = ['delay', 'month', 'day', 'dow', 'hour', 'distance', 'carrier', 'dest', 'days_from_holiday']
col_types = {'delay': int, 'month': int, 'day': int, 'dow': int, 'hour': int, 'distance': int,
             'carrier': str, 'dest': str, 'days_from_holiday': int}
data_2007 = readfile.read_csv_from_hdfs('airline/fm/ord_2007_1', cols, col_types)
data_2008 = readfile.read_csv_from_hdfs('airline/fm/ord_2008_1', cols, col_types)

# Create training set and test set
cols = ['month', 'day', 'dow', 'hour', 'distance', 'days_from_holiday']
train_y = data_2007['delay'] >= 15
train_x = data_2007[cols]

test_y = data_2008['delay'] >= 15
test_x = data_2008[cols]

print(train_x.shape)

# Create logistic regression model with L2 regularization
clf_lr = linear_model.LogisticRegression(penalty='l2', class_weight='auto')
clf_lr.fit(train_x, train_y)

# Predict output labels on test set
pr = clf_lr.predict(test_x)

# display evaluation metrics
cm = confusion_matrix(test_y, pr)
print("Confusion matrix")
print(pd.DataFrame(cm))
report_lr = precision_recall_fscore_support(list(test_y), list(pr), average='micro')
print("\nprecision = %0.2f, recall = %0.2f, F1 = %0.2f, accuracy = %0.2f\n" % \
      (report_lr[0], report_lr[1], report_lr[2], accuracy_score(list(test_y), list(pr))))

# read files
cols = ['delay', 'month', 'day', 'dow', 'hour', 'distance', 'carrier', 'dest', 'days_from_holiday']
col_types = {'delay': int, 'month': int, 'day': int, 'dow': int, 'hour': int, 'distance': int,
             'carrier': str, 'dest': str, 'days_from_holiday': int}
data_2007 = readfile.read_csv_from_hdfs('airline/fm/ord_2007_1', cols, col_types)
data_2008 = readfile.read_csv_from_hdfs('airline/fm/ord_2008_1', cols, col_types)

# Create training set and test set
train_y = data_2007['delay'] >= 15
categ = [cols.index(x) for x in ('hour', 'month', 'day', 'dow', 'carrier', 'dest')]
enc = OneHotEncoder(categorical_features=categ)
df = data_2007.drop('delay', axis=1)
df['carrier'] = pd.factorize(df['carrier'])[0]
df['dest'] = pd.factorize(df['dest'])[0]
train_x = enc.fit_transform(df)

test_y = data_2008['delay'] >= 15
df = data_2008.drop('delay', axis=1)
df['carrier'] = pd.factorize(df['carrier'])[0]
df['dest'] = pd.factorize(df['dest'])[0]
test_x = enc.transform(df)

print(train_x.shape)

# Create Random Forest classifier with 50 trees
clf_rf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
clf_rf.fit(train_x.toarray(), train_y)

# Evaluate on test set
pr = clf_rf.predict(test_x.toarray())

# print results
cm = confusion_matrix(test_y, pr)
print("Confusion matrix")
print(pd.DataFrame(cm))
report_svm = precision_recall_fscore_support(list(test_y), list(pr), average='micro')
print("\nprecision = %0.2f, recall = %0.2f, F1 = %0.2f, accuracy = %0.2f\n" % \
      (report_svm[0], report_svm[1], report_svm[2], accuracy_score(list(test_y), list(pr))))


# Convert Celsius to Fahrenheit
def fahrenheit(x): return x * 1.8 + 32.0


# read files
cols = ['delay', 'month', 'day', 'dow', 'hour', 'distance', 'carrier', 'dest', 'days_from_holiday',
        'origin_tmin', 'origin_tmax', 'origin_prcp', 'origin_snow', 'origin_wind']
col_types = {'delay': int, 'month': int, 'day': int, 'dow': int, 'hour': int, 'distance': int,
             'carrier': str, 'dest': str, 'days_from_holiday': int,
             'origin_tmin': float, 'origin_tmax': float, 'origin_prcp': float, 'origin_snow': float,
             'origin_wind': float}

data_2007 = readfile.read_csv_from_hdfs('airline/fm/ord_2007_2', cols, col_types)
data_2008 = readfile.read_csv_from_hdfs('airline/fm/ord_2008_2', cols, col_types)

data_2007['origin_tmin'] = data_2007['origin_tmin'].apply(lambda x: fahrenheit(x / 10.0))
data_2007['origin_tmax'] = data_2007['origin_tmax'].apply(lambda x: fahrenheit(x / 10.0))
data_2008['origin_tmin'] = data_2008['origin_tmin'].apply(lambda x: fahrenheit(x / 10.0))
data_2008['origin_tmax'] = data_2008['origin_tmax'].apply(lambda x: fahrenheit(x / 10.0))

# Create training set and test set
train_y = data_2007['delay'] >= 15
categ = [cols.index(x) for x in ('hour', 'month', 'day', 'dow', 'carrier', 'dest')]
enc = OneHotEncoder(categorical_features=categ)
df = data_2007.drop('delay', axis=1)
df['carrier'] = pd.factorize(df['carrier'])[0]
df['dest'] = pd.factorize(df['dest'])[0]
train_x = enc.fit_transform(df)

test_y = data_2008['delay'] >= 15
df = data_2008.drop('delay', axis=1)
df['carrier'] = pd.factorize(df['carrier'])[0]
df['dest'] = pd.factorize(df['dest'])[0]
test_x = enc.transform(df)

print(train_x.shape)

# Create Random Forest classifier with 100 trees
clf_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
clf_rf.fit(train_x.toarray(), train_y)

# Evaluate on test set
pr = clf_rf.predict(test_x.toarray())

# print results
cm = confusion_matrix(test_y, pr)
print("Confusion matrix")
print(pd.DataFrame(cm))
report_rf = precision_recall_fscore_support(list(test_y), list(pr), average='micro')
print("precision = %0.2f, recall = %0.2f, F1 = %0.2f, accuracy = %0.2f\n" % \
      (report_rf[0], report_rf[1], report_rf[2], accuracy_score(list(test_y), list(pr))))
