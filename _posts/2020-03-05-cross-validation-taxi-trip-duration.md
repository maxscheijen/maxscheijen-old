---
layout: post
title: "Robust Cross Validation for Data Science - Taxi Trip Duration"
categories: [data-science, machine-learning, cross-validation]
author: Max Scheijen
---

One of the first things you learn when applying machine learning models is the notion of cross-validation. Training a model, which basically is learning the parameters of a prediction function, and evaluating the performance of a model on the same dataset is a methodological mistake. The machine learning model could learn the labels of the data and reproduce them. However, this is not really what we want. If we then deploy the machine learning model on unseen data, we run the risk of predicting not anything useful. Our model has only learned the training data.

In this post, I present a robust cross-validation method for tabular data. This method, among other things, can be used in Data Science competitions.

To demonstrate this technique, we use the <a href="https://www.kaggle.com/c/nyc-taxi-trip-duration/overview" target="_blank">New York City Taxi Trip Duration</a> competition hosted on <a href="https://www.kaggle.com/" target="_blank">Kaggle</a>. In this competition, we try to predict the trip duration.

This post is not intended to achieve a high score in the competition in question. I  demonstrate a way to implement a cross-validation method that gives the same results on validation data as on unseen test data.

<div class="alert" style="background-color: #f6f6f6">
  <strong>Note: </strong> This methods is heavily based on <a href="https://twitter.com/abhi1thakur" target="_blank">Abhishek Thakur</a> cross validation method.
</div>

```python
import pandas as pd

# load data
train = pd.read_csv('data/train.csv').drop(['id', 'dropoff_datetime'], axis=1)
test = pd.read_csv('data/test.csv').drop(['id'], axis=1)

# display first 3 rows
train.head(3)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>vendor_id</th>
      <th>pickup_datetime</th>
      <th>passenger_count</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>store_and_fwd_flag</th>
      <th>trip_duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>2016-03-14 17:24:55</td>
      <td>1</td>
      <td>-73.982155</td>
      <td>40.767937</td>
      <td>-73.964630</td>
      <td>40.765602</td>
      <td>N</td>
      <td>455</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2016-06-12 00:43:35</td>
      <td>1</td>
      <td>-73.980415</td>
      <td>40.738564</td>
      <td>-73.999481</td>
      <td>40.731152</td>
      <td>N</td>
      <td>663</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2016-01-19 11:35:24</td>
      <td>1</td>
      <td>-73.979027</td>
      <td>40.763939</td>
      <td>-74.005333</td>
      <td>40.710087</td>
      <td>N</td>
      <td>2124</td>
    </tr>
  </tbody>
</table>
</div>

## $k$-fold cross-validation

We load the train and test data into the cell above. I drop the `dropoff_datetime` because this feature isn't present in the testing data.

We use $k$-fold cross-validation to implement our cross-validation method. In $k$-fold cross-validation we estimate what our model has learned based on new data that the model has not yet seen. The $k$ in $k$-fold cross-validation shows how many groups we divide our data into. If $k$ = 5 then we divide our data into 5 groups. We then train our algorithm on 4 groups of the data, and we validate the model on the 1 group of data that the model has not yet seen. We repeat this process until we have trained our model on each group of data once. Also, each group of data has been used to validate our model.

To implement this method we use `sklearn`'s $k$-fold cross-validation capabilities. In our case, we are dealing with a regression problem. This is why we use a simple k-fold method. However, if you are dealing with a classification problem you can use stratified $k$-fold cross-validation. This ensures that the frequency of labels between the folds is the same. And increases ensures better consistency between validation and testing scores.

```python
from sklearn.model_selection import KFold

# number of folds used
FOLDS = 5

# instantiate k-fold cross-validation
kfold = KFold(n_splits=FOLDS, shuffle=False)
```

In this case choose $k$ = 5. We have a relatively large data set, so it seems to me that 5 folds will generate a robust validation score. We are now going to divide our data into 5 groups/folds. We create a new column in which we record the fold the observation belongs to.

However, before we do this, we randomize the order of the data. This way, we prevent the order in which the data is placed from becoming important, which can incorporate bias in our validation score.

<div class="alert" style="background-color: #f6f6f6">
  <strong>Note: </strong>  don't do this with a time series. Use sklearn's dedicated time series cross validation functionality.
</div>

```python
# randomize order of dataframe
train = train.sample(frac=1, random_state=1).reset_index(drop=True)

# create fold columns and store the fold
for fold, (train_idx, valid_idx) in enumerate(kfold.split(X=train, y=train['trip_duration'])):
    train.loc[valid_idx, 'kfold'] = fold

# display rows
train.head(3)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>vendor_id</th>
      <th>pickup_datetime</th>
      <th>passenger_count</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>store_and_fwd_flag</th>
      <th>trip_duration</th>
      <th>kfold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>2016-02-27 20:13:05</td>
      <td>1</td>
      <td>-73.981728</td>
      <td>40.749500</td>
      <td>-73.945915</td>
      <td>40.792061</td>
      <td>N</td>
      <td>692</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2016-06-04 09:54:05</td>
      <td>1</td>
      <td>-73.979088</td>
      <td>40.771606</td>
      <td>-73.946518</td>
      <td>40.822655</td>
      <td>N</td>
      <td>990</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2016-05-06 17:40:05</td>
      <td>1</td>
      <td>-73.989700</td>
      <td>40.738651</td>
      <td>-73.997772</td>
      <td>40.754051</td>
      <td>N</td>
      <td>647</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

Now we have created a column in which we record the fold the observation belongs to. If we look at the size distribution of the number of folds, we see that the folds are approximately the same. This is what we want!

Sometimes there can be a small difference in the number of observations (in our case!) because the number of recorded samples is not easily divisible by the number of folds.

```python
# size of every fold
train['kfold'].value_counts()
```

```shell
1.0    291729
3.0    291729
2.0    291729
0.0    291729
4.0    291728
Name: kfold, dtype: int64
```

Even though this post does not focus on creating a well-performing predictive model, I'll do simple feature engineering to make the performance of our model a little bit better. This makes me feel a little bit better about myself! The function below extracts some simple date features from the date feature. This also gives us some more variables to work with.

```python
def extract_dates(data, date_column):
    # get datetime index
    d = pd.DatetimeIndex(data[date_column])

    # create date features
    data['year'], data['month'], data['week'], data['dayofweek'], data['hour'], data['minute'] =\
    d.year, d.month, d.week, d.dayofweek, d.hour, d.minute
    return data.drop([date_column], axis=1)

# extract date features
train = extract_dates(train, 'pickup_datetime')

# display
train.head(3)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>vendor_id</th>
      <th>passenger_count</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>store_and_fwd_flag</th>
      <th>trip_duration</th>
      <th>kfold</th>
      <th>year</th>
      <th>month</th>
      <th>week</th>
      <th>dayofweek</th>
      <th>hour</th>
      <th>minute</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>-73.981728</td>
      <td>40.749500</td>
      <td>-73.945915</td>
      <td>40.792061</td>
      <td>N</td>
      <td>692</td>
      <td>0.0</td>
      <td>2016</td>
      <td>2</td>
      <td>8</td>
      <td>5</td>
      <td>20</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>-73.979088</td>
      <td>40.771606</td>
      <td>-73.946518</td>
      <td>40.822655</td>
      <td>N</td>
      <td>990</td>
      <td>0.0</td>
      <td>2016</td>
      <td>6</td>
      <td>22</td>
      <td>5</td>
      <td>9</td>
      <td>54</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>-73.989700</td>
      <td>40.738651</td>
      <td>-73.997772</td>
      <td>40.754051</td>
      <td>N</td>
      <td>647</td>
      <td>0.0</td>
      <td>2016</td>
      <td>5</td>
      <td>18</td>
      <td>4</td>
      <td>17</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>

## Model training using $k$-folds

We are now switching to training our machine learning model on our 5 folds. This means that we get as many models as folds, in this case, we'll end up with 5 different models.

We create a function that allows us to grab a specific fold of the training data set. This will be our validation dataset! Then the remaining folds become our training set on which we'll train the model. It also lets us define the fold we want to validate our model on.  Furthermore, we can choose to drop certain variables. We always want to remove the `k-fold` variable because this feature is not present in the testing data. This variable does not provide any predictive information.

As an example of the validation and and training fold:

```shell
FOLD = 0 - VALIDATION FOLD = 0 - TRAINING FOLD = [1, 2, 3, 4]
FOLD = 1 - VALIDATION FOLD = 1 - TRAINING FOLD = [0, 2, 3, 4]
FOLD = 2 - VALIDATION FOLD = 2 - TRAINING FOLD = [1, 0, 3, 4]
FOLD = 3 - VALIDATION FOLD = 3 - TRAINING FOLD = [1, 2, 0, 4]
FOLD = 4 - VALIDATION FOLD = 4 - TRAINING FOLD = [1, 2, 3, 0]
```

We then return our dataset with training features (`X_train`), validation features (`X_valid`), our training target values (`y_train`), and our validation targets (`y_valid`).

__NOTE__: In this case, I log-transform the target feature for better predictive performance. This is problem specific.

```python
def get_folds(data, target, drop_features, fold):

    # get training folds and validation fold
    train = data[data.kfold != fold].reset_index(drop=True)
    valid = data[data.kfold == fold].reset_index(drop=True)

    # extract targets
    y_train = np.log1p(train[target])
    y_valid = np.log1p(valid[target])

    # features that need to be dropped
    feat_to_drop  = [target] + drop_features

    # drop features in train data
    X_train = train.drop(feat_to_drop, axis=1)

    # make validation features equal to train features
    X_valid = valid[X_train.columns]

    return X_train, X_valid, y_train, y_valid  
```

As a demonstration, we use a random forest machine learning model to predict our target variable. However, you can use any machine learning model here. I set a random state for reproducibility.

```python
from sklearn.ensemble import RandomForestRegressor

MODEL = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
```

We then create a new function that takes a model and the number of folds. The model is then trained on the training folds data and validated on the validation folds. For every fold, we save the trained model as a pickle file in the models' directory.

We also store names of the features on which the model is trained. We can use these features later to select variables from the test set. When there are categorical variables in your dataset, or you need to impute missing data,  also save this transformation based on the folds. This ensures that we can apply the same transformations to the train, validation, and testing data. 

```python
import joblib
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

def train_folds(data, folds, model):

    scores = []

    # loop over number of folds
    for fold in range(folds):

        # get train and validation folds
        X_train, X_valid, y_train, y_valid = get_folds(data, 'trip_duration', ['kfold'], fold)

        # fitting label encoding on train, validation and testing data
        enc_feature = 'store_and_fwd_flag'

        le = LabelEncoder()

        le.fit(np.concatenate([X_train[enc_feature], X_valid[enc_feature], test[enc_feature]]))

        X_train[enc_feature] = le.transform(X_train[enc_feature])
        X_valid[enc_feature] = le.transform(X_valid[enc_feature])

        # train on train folds
        m = model
        m.fit(X_train, y_train)

        # get prediction on valid fold
        valid_pred = m.predict(X_valid)

        # print score metric
        valid_score = np.sqrt(metrics.mean_squared_error(y_valid, valid_pred))
        scores.append(valid_score)

        # print fold model score
        print(f"FOLD: {fold} - RMSLE: {round(valid_score, 4)}")

        # save model, features and label encoding
        joblib.dump(m, f"models/MODEL_{fold}.json", compress=3)
        joblib.dump(X_train.columns.values, f"models/FEAT_{fold}.json", compress=3)
        joblib.dump(le, f"models/ENC_{fold}.json", compress=3)

    # print mean score
    print(f"\nMean RMSLE: {np.mean(scores):.4f} ({np.std(scores):.4f})")

train_folds(train, FOLDS, MODEL)
```

```shell
FOLD: 0 - RMSLE: 0.432
FOLD: 1 - RMSLE: 0.4351
FOLD: 2 - RMSLE: 0.4286
FOLD: 3 - RMSLE: 0.4363
FOLD: 4 - RMSLE: 0.4326

Mean RMSLE: 0.4329 (0.0027)
```

## Making predictions using folds

We can now use the 5 models that we trained and saved to make predictions about our test data. We can then take the average of these 5 predictions and use this as our final prediction.

Before we do this, we first create a new data frame with 5 columns to store our 5 test data predictions.

```python
# generate column names
folds_columns = [f"FOLD_{i}" for i in range(FOLDS)]

# create empty dataframe
pred_df = pd.DataFrame(data=np.zeros((len(test), FOLDS)), columns=folds_columns)

# display empty dataframe
pred_df.head(3)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FOLD_0</th>
      <th>FOLD_1</th>
      <th>FOLD_2</th>
      <th>FOLD_3</th>
      <th>FOLD_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

We now loop over every fold and load the model, the features files, and the encoding that corresponds to the specific fold. Each of these models then makes a prediction on the test data columns after apply label encoding. We store this prediction in the relevant column of the new data frame.

```python
# loop over folds
for fold in range(FOLDS):

    # load test data
    test = pd.read_csv('data/test.csv').drop(['id'], axis=1)

    # create date features
    test = extract_dates(test, 'pickup_datetime')

    # load label encoder and transform test feature
    encoder = joblib.load(f'models/ENC_{fold}.json')

    test['store_and_fwd_flag'] = encoder.transform(test['store_and_fwd_flag'])

    # load fold model
    m = joblib.load(f'models/MODEL_{fold}.json')

    # load fold features
    feat = joblib.load(f'models/FEAT_{fold}.json')

    # predict on test data
    pred = m.predict(test[feat])

    # store predictions in fold column
    pred_df[f"FOLD_{fold}"] = np.expm1(pred)

pred_df.head(3)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FOLD_0</th>
      <th>FOLD_1</th>
      <th>FOLD_2</th>
      <th>FOLD_3</th>
      <th>FOLD_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>628.423460</td>
      <td>716.639654</td>
      <td>815.767972</td>
      <td>728.653350</td>
      <td>883.245895</td>
    </tr>
    <tr>
      <th>1</th>
      <td>553.966449</td>
      <td>639.652247</td>
      <td>622.347987</td>
      <td>668.222903</td>
      <td>671.115124</td>
    </tr>
    <tr>
      <th>2</th>
      <td>672.211223</td>
      <td>510.996218</td>
      <td>563.263305</td>
      <td>395.760266</td>
      <td>506.161285</td>
    </tr>
  </tbody>
</table>
</div>

We can then take the average prediction of these five models as our final prediction.

```python
# get the mean of k predictions
final_preds = pred_df.mean(axis=1)

# display first 3 rows
final_preds.head(3)
```

```shell
0    754.546066
1    631.060942
2    529.678459
dtype: float64
```

We store this average prediction in our submission file and upload it to Kaggle to get our testing score. Our testing score on Kaggle is __0.41885__. The internal mean validation score was __0.4329__, which isn't too bad of a difference.

```python
# load sample submission
sample_sub = pd.read_csv('data/sample_submission.csv')

# store predictions in column
sample_sub['trip_duration'] = final_preds

# save predictions
sample_sub.to_csv('submissions/submission.csv', index=None)
```

## Conclusion

In this post, I demonstrated a simple way to use cross-validation to train a model on folds and use those folds to predict on unseen test data. This results in a cross-validation and testing score that are relatively close to each other.
