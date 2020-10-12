---
layout: post
title: "Quick and Dirty Machine Learning: Modeling Earthquake Damage"
categories: [data-science, machine-learning]
author: Max Scheijen
---

In this post, I would like to demonstrate a quick way to create a good performing machine learning model. It is often encouraging to get a good performing model fast, without major pre-processing and featuring engineering. After this, we can use model selection, hyper-parameter tweaking, and feature engineering to squeeze out a little more predictive performance. However, it is really encouraging to get a well-performing fast.

So let's get started! I will be using data from a Data Science competition to asses the model's performance. The contest is called <a href="https://www.drivendata.org/competitions/57/nepal-earthquake/">"Richter's Predictor: Modeling Earthquake Damage"</a> hosted by <a href="https://www.drivendata.org/" target="_blank">DrivenData</a>. The goal is to predict the damage to buildings caused by an earthquake. The amount of damage is divided into 3 categories:  (1) low, (2) medium, (3) almost complete destruction.

In the cell below, we load the train values, train labels, test values, and the sample submission file and display the first couple of rows.


```python
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
```


```python
train_labels = pd.read_csv("data/train_labels.csv").drop('building_id', axis=1)
train_values = pd.read_csv("data/train_values.csv").drop('building_id', axis=1)

test_values = pd.read_csv("data/test.csv").drop('building_id', axis=1)
sample_sub = pd.read_csv("data/submission_format.csv")

train_values.head(2)
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
      <th>geo_level_1_id</th>
      <th>geo_level_2_id</th>
      <th>geo_level_3_id</th>
      <th>count_floors_pre_eq</th>
      <th>age</th>
      <th>area_percentage</th>
      <th>height_percentage</th>
      <th>land_surface_condition</th>
      <th>foundation_type</th>
      <th>roof_type</th>
      <th>ground_floor_type</th>
      <th>other_floor_type</th>
      <th>position</th>
      <th>plan_configuration</th>
      <th>has_superstructure_adobe_mud</th>
      <th>has_superstructure_mud_mortar_stone</th>
      <th>has_superstructure_stone_flag</th>
      <th>has_superstructure_cement_mortar_stone</th>
      <th>has_superstructure_mud_mortar_brick</th>
      <th>has_superstructure_cement_mortar_brick</th>
      <th>has_superstructure_timber</th>
      <th>has_superstructure_bamboo</th>
      <th>has_superstructure_rc_non_engineered</th>
      <th>has_superstructure_rc_engineered</th>
      <th>has_superstructure_other</th>
      <th>legal_ownership_status</th>
      <th>count_families</th>
      <th>has_secondary_use</th>
      <th>has_secondary_use_agriculture</th>
      <th>has_secondary_use_hotel</th>
      <th>has_secondary_use_rental</th>
      <th>has_secondary_use_institution</th>
      <th>has_secondary_use_school</th>
      <th>has_secondary_use_industry</th>
      <th>has_secondary_use_health_post</th>
      <th>has_secondary_use_gov_office</th>
      <th>has_secondary_use_use_police</th>
      <th>has_secondary_use_other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>487</td>
      <td>12198</td>
      <td>2</td>
      <td>30</td>
      <td>6</td>
      <td>5</td>
      <td>t</td>
      <td>r</td>
      <td>n</td>
      <td>f</td>
      <td>q</td>
      <td>t</td>
      <td>d</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>v</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>900</td>
      <td>2812</td>
      <td>2</td>
      <td>10</td>
      <td>8</td>
      <td>7</td>
      <td>o</td>
      <td>r</td>
      <td>n</td>
      <td>x</td>
      <td>q</td>
      <td>s</td>
      <td>d</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>v</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Distribution of classes

One of the first things you should when dealing with classification is to look at the distribution of classes. For this, I use pandas value_count capabilities to get the frequency distribution of the target labels. Not all target categories are equally present. 

Label 2 is the majority class, with 56% of the observations. This percentage will be our baseline score. We need to outperform this score for our model to add something valuable. Otherwise, we would be better of always predicting label 2. 


```python
train_labels.damage_grade.value_counts(normalize=True).sort_index()
```




    1    0.096408
    2    0.568912
    3    0.334680
    Name: damage_grade, dtype: float64



## Preprocessing

The only preprocessing we'll do is to get the data ready for the machine learning algorithm.

### Missing data

Often machine learning models don't allow for missing values in our data. We should identify these values and treat them. In our specific case, we are lucky because we don't have any missing values. However, if we would encounter them, these values need to be replaced. 

A good starting point is replacing missing numerical values with the mean or median, and replacing missing categorical values with the most frequent (mode) value. Note that your method of imputing missing values can have an impact on the performance of your model.


```python
train_values.isna().sum().sum(), test_values.isna().sum().sum()
```




    (0, 0)



### Categorical features

Most of the time, machine learning models also don't allow for categorical variables. We need to identify these features and encode them. One important note is that there should be consistency between the numerical representation of these values in the features in the train and test set.  

This can be achieved by first concatenating the test data at the end of the train data. Secondly, we encode the categorical features, after which we split the data back into a train and test set. We now have a constant representation of these variables between our train and test set.

Label encoding is a good starting point. This method encodes features with a value between 0 and n_classes-1. This technique works best with ordinal features because it implies ordering. However, because this post is about getting a quick model done, we apply it to all categorical features. We use some assert statements to test if the shape of the original train and test set are the same as the encoded ones.


```python
from sklearn.preprocessing import LabelEncoder

all_data = pd.concat([train_values, test_values])
categorical_features = all_data.select_dtypes('object').columns

label_encoder = LabelEncoder()
all_data[categorical_features] = all_data[categorical_features].apply(label_encoder.fit_transform)

train_enc = all_data[:len(train_values)]
test_enc = all_data[len(train_values):]

assert train_enc.shape == train_values.shape 
assert test_enc.shape == test_values.shape

train_enc.head(3)
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
      <th>geo_level_1_id</th>
      <th>geo_level_2_id</th>
      <th>geo_level_3_id</th>
      <th>count_floors_pre_eq</th>
      <th>age</th>
      <th>area_percentage</th>
      <th>height_percentage</th>
      <th>land_surface_condition</th>
      <th>foundation_type</th>
      <th>roof_type</th>
      <th>ground_floor_type</th>
      <th>other_floor_type</th>
      <th>position</th>
      <th>plan_configuration</th>
      <th>has_superstructure_adobe_mud</th>
      <th>has_superstructure_mud_mortar_stone</th>
      <th>has_superstructure_stone_flag</th>
      <th>has_superstructure_cement_mortar_stone</th>
      <th>has_superstructure_mud_mortar_brick</th>
      <th>has_superstructure_cement_mortar_brick</th>
      <th>has_superstructure_timber</th>
      <th>has_superstructure_bamboo</th>
      <th>has_superstructure_rc_non_engineered</th>
      <th>has_superstructure_rc_engineered</th>
      <th>has_superstructure_other</th>
      <th>legal_ownership_status</th>
      <th>count_families</th>
      <th>has_secondary_use</th>
      <th>has_secondary_use_agriculture</th>
      <th>has_secondary_use_hotel</th>
      <th>has_secondary_use_rental</th>
      <th>has_secondary_use_institution</th>
      <th>has_secondary_use_school</th>
      <th>has_secondary_use_industry</th>
      <th>has_secondary_use_health_post</th>
      <th>has_secondary_use_gov_office</th>
      <th>has_secondary_use_use_police</th>
      <th>has_secondary_use_other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>487</td>
      <td>12198</td>
      <td>2</td>
      <td>30</td>
      <td>6</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>900</td>
      <td>2812</td>
      <td>2</td>
      <td>10</td>
      <td>8</td>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21</td>
      <td>363</td>
      <td>8973</td>
      <td>2</td>
      <td>10</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Cross-validation

Before training our model, we should create a validation dataset. This lets us internally test our model, before predicting the test data. We're dealing with a classification problem, with an uneven class distribution. Therefore we should us stratification when splitting our data into a train and validation sets. This ensures that we have the same class distribution in our train and validation set.


```python
from sklearn.model_selection import train_test_split

X = train_enc
y = train_labels

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
```

## Modeling the Data

Now that this is all done, we can start training our machine learning model. We use a LightGBM, which is a gradient boosting framework that uses tree-based learning algorithms. According to the developers, it is designed to be distributed and efficient with the following advantages:

* Faster training speed and higher efficiency.
* Lower memory usage.
* Better accuracy.
* Support for parallel and GPU learning.
* Capable of handling large-scale data.

A basic LightGBM model can be trained in the same way as you fit a scikit-learn machine learning model. We start with 200 estimators, max depth of 50 with 900 leaves. This model only takes about a minute to train.


```python
%%time
import lightgbm as lgb
model = lgb.LGBMClassifier(random_state=1, n_estimators=200, max_depth=50, num_leaves=900)
model.fit(X_train, y_train)
preds = model.predict(X_valid)
```

    CPU times: user 3min 7s, sys: 2.76 s, total: 3min 10s
    Wall time: 51.2 s


## Evaluating the model

To evaluate the performance of our model, we use the **micro averaged f1 score**, the same as in the contest. Our model gives us an f1-micro averaged score of **0.7422**, which is quite good. Especially when you keep in mind that we did so little processing and feature engineering.


```python
from sklearn.metrics import f1_score

score_model = f1_score(y_valid, preds, average='micro')
round(score_model, 4)
```




    0.7422



The model does outperform the naive majority class which gets a micro f1-score of **0.5689**.


```python
naive_pred = np.zeros((len(y_valid))) + 2
score_naive = f1_score(y_valid, naive_pred, average='micro')
round(score_naive, 4)
```




    0.5689



## Training the full model and predicting on test data

We can now train on the full dataset, which also lets us train on the validation set. After this, we can make predictions on the test dataset.


```python
%%time
model.fit(X, y)
test_preds = model.predict(test_enc)
```

    CPU times: user 3min 45s, sys: 3.1 s, total: 3min 48s
    Wall time: 1min 1s


We store the test dataset predictions in the damage_grade columns of the sample submission file and save it as a CSV.


```python
sample_sub['damage_grade'] = test_preds
sample_sub.to_csv("prediction.csv", index=None)
```

## Conclusion

After submitting to the competition, we a score of **0.7446**. As of 2020-03-18, this ranks us in the top **6-7%** of the leaderboard. This post demonstrated that we can achieve quite good performing models with almost no preprocessing, by leveraging the power of the LightGBM machine learning framework.
