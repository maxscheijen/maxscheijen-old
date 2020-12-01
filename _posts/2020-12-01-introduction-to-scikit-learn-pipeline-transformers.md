---
layout: post
title: "Why you should be using of Scikit-learn Pipelines"
categories: [scikit-learn, pipelines, production, scikit-learn]
author: Max Scheijen
---

In my opinion, pipelines in scikit-learn are one of the most useful things in the library. They allow you to sequentially apply a list of transformers and a final estimator to your data. Furthermore, they can be used in cross-validation. This ensures that data transformations within the cross-validation loop are only fitted on the training data. This prevents data leakage and better generalization on the test dataset or in production. Pipelines can also be used in a  grid or random search for the best hyperparameters. However, this is not limited to model parameters. We can also search for the best data transformations hyper-parameters (imputation with the median or the mean) in conjunction with the model hyper-parameters. 

However, in this post, I look at the fundamentals of the pipeline. We focus on implementing scikit-learn transformers and custom transformers into a scikit-learn pipeline. How do we apply different transformations to different datatypes or features? After implementing this, we can save the entire data pipeline (data transformations and the estimator) into a single artifact. This is great when we want to use the model in production because we need to deal with one file.

We use a house pricing <a href="https://www.kaggle.com/c/house-prices-advanced-regression-techniques" target="_blank">dataset</a> from kaggle for demonstrating the

```python
import pandas as pd

from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("train.csv")

# Features and target select
X = data.drop(columns=["SalePrice"])
y = data["SalePrice"]

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=1)

# Display first two rows of features
X_train.head(1)
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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>922</td>
      <td>90</td>
      <td>RL</td>
      <td>67.0</td>
      <td>8777</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
    </tr>
  </tbody>
</table>
</div>

## Pipelines

Scikit-learn's pipeline object expects a list of steps. These steps need to be tuples containing a name and a scikit-learn transformer (name, transform). Let's first create a simple pipeline that fills missing numerical values with the mean. This can be done with the `SimpleImputer()` class from scikit-learn. 

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Numerical single transformer pipelinen
num_pipeline = Pipeline(steps=[
  ("num_imputer",  SimpleImputer(strategy="mean")),
])

# Fit numerical transformer pipeline
num_pipeline.fit(X_train[["LotFrontage"]])

# Display first five transformed rows
num_pipeline.transform(X_test[["LotFrontage"]])[:5]
```

```shell
array([[80.        ],
       [60.        ],
       [70.21063608],
       [21.        ],
       [70.21063608]])
```

In the preceding code block, we created a pipeline that imputes numerical values and fills them with the mean. The pipeline object can be used to fit and transform data just as you are used to when using a single transformer.

Let us now chain two transformers together into a single pipeline object. For example, after imputing the missing data and filling it with the mean, we want to scale the data between 0 and 1. We can use the `MinMaxScaler()` to do this scaling.

```python
from sklearn.preprocessing import MinMaxScaler

# Numerical multi transformer pipeline
num_pipeline = Pipeline(steps=[
  ("num_imputer",  SimpleImputer(strategy="mean")),
  ("scaler", MinMaxScaler(feature_range=(0, 1)))
])

# Fit numerical transformer pipeline
num_pipeline.fit(X_train[["LotFrontage"]])

# Display first five transformed rows
num_pipeline.transform(X_test[["LotFrontage"]])[:5]
```

```shell
array([[0.20205479],
       [0.13356164],
       [0.16852958],
       [0.        ],
       [0.16852958]])
```

We have now chained to transformers together and can be fitted on the training data and transform testing data. We can also serialize this entire pipeline if we want to use it later in production.

## Datatypes and Pipeline Transformers

Until this point, we only transformed numerical data. How do we deal with a table containing columns with numerical values and columns containing categorical data? Do we need several separate pipelines? This can become complicated quickly. However, scikit-learn provides a nice and simple solution to this problem: the column transformer. This transformer can apply different preprocessing and feature extraction pipelines to different subsets of features. Meaning that we can use other preprocessing steps to transform the numerical data than the transformations used to process the categorical data. 

So how do we do this? First, we create a pipeline for numerical data transformers. These transformers will be applied linearly (the order that the transformation steps are implemented). Secondly, we create a transformation pipeline for categorical data. We can now use the ColumnTransformer(), which takes a list of transformers as tuples (name, transformer pipeline, and the columns to transform) and apply the transformations.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Pipeline of numerical transformers
num_transformer_pipeline = Pipeline(steps=[
  ("num_imputer",  SimpleImputer(strategy="mean")),
  ("scaler", MinMaxScaler(feature_range=(0, 1)))
])

# Pipeline of categorical transformers
cat_transformer_pipeline = Pipeline(steps=[
  ("cat_imputer", SimpleImputer(strategy="most_frequent")),
  ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Create preprocessing pipeline
preprocessors = ColumnTransformer(transformers=[
  ("num_transformer", num_transformer_pipeline, ["LotFrontage"]),
  ("cat_transformer", cat_transformer_pipeline, ["Street", "Alley"]),
])

# Fit transformer pipeline
preprocessors.fit(X_train)

# Display first three transformed rows
preprocessors.transform(X_test)[:3, :]
```

```shell
array([[0.20205479, 0., 1., 1., 0.],
       [0.13356164, 0., 1., 1., 0.],
       [0.16852958, 0., 1., 1., 0.]])
```

As you can see in the diagram, we have two transformer pipelines that are combined into a single pipeline object. The numerical transformer pipeline is applied to the `LotFrontage` column of the table. In parallel, the categorical transformer pipeline is applied to the `Street` and `Alley` columns. Basically, we can create data processing pipelines for every individual feature in our dataset.

However, we do not necessarily need to specify the columns to transform. We can select them automatically using `make_column_selector()`. This class can select columns based on the data type or the column name.

```python
from sklearn.compose import make_column_selector

# Create preprocessing pipeline
preprocessors = ColumnTransformer(transformers=[
  ("num_transformer", num_transformer_pipeline, make_column_selector(dtype_include="number")),
  ("cat_transformer", cat_transformer_pipeline, make_column_selector(dtype_include="category")),
])

# Fit transformer pipeline
preprocessors.fit(X_train)

# Display first transformed row
preprocessors.transform(X_test)[0, :]
```

```shell
array([0.17683345, 0.23529412, 0.20205479, 0.05204609, 0.66666667,
       0.42857143, 0.93478261, 0.85      , 0.1075    , 0.06396173,
       0.        , 0.25770548, 0.15761047, 0.14433226, 0.44284188,
       0.        , 0.27467973, 0.        , 0.        , 0.66666667,
       0.5       , 0.375     , 0.5       , 0.41666667, 0.33333333,
       0.91818182, 0.5       , 0.3977433 , 0.        , 0.17550274,
       0.        , 0.48228346, 0.        , 0.        , 0.        ,
       0.36363636, 0.5       ])
```

In the preceding code block, we use the make_column_selector to only include select all the numerical ("number") columns and then to select all the categorical columns ("category"). This automates selecting the columns and applying the different transformations to the data. This single pipeline can be serialized. This makes it easy to use data transformers in production because we only need to load the preprocessing pipeline to call `transform()` to process the new data.

To complete the pipeline, you can add an estimator to the end of the pipeline. This is done by creating another pipeline instance where you combine the data processing pipeline with an estimator.

```python
from sklearn.linear_model import LinearRegression

# Preprocessing transformers and estimator
regressor_pipeline = Pipeline(steps=[
  ("preprocessors", preprocessors),
  ("regressor", LinearRegression())
])

# Fit full pipeline
regressor_pipeline.fit(X_train, y_train)
```

<style>div.sk-top-container {color: black;background-color: white;}div.sk-toggleable {background-color: white;}label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.2em 0.3em;box-sizing: border-box;text-align: center;}div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}div.sk-estimator {font-family: monospace;background-color: #f0f8ff;margin: 0.25em 0.25em;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;}div.sk-estimator:hover {background-color: #d4ebff;}div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;}div.sk-item {z-index: 1;}div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}div.sk-parallel-item:only-child::after {width: 0;}div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0.2em;box-sizing: border-box;padding-bottom: 0.1em;background-color: white;position: relative;}div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}div.sk-label-container {position: relative;z-index: 2;text-align: center;}div.sk-container {display: inline-block;position: relative;}</style><div class="sk-top-container"><div class="sk-container"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="3ada28d2-9f9d-4747-b84b-2b3935a05b09" type="checkbox" ><label class="sk-toggleable__label" for="3ada28d2-9f9d-4747-b84b-2b3935a05b09">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[('preprocessors',
                 ColumnTransformer(transformers=[('num_transformer',
                                                  Pipeline(steps=[('num_imputer',
                                                                   SimpleImputer()),
                                                                  ('scaler',
                                                                   MinMaxScaler())]),
                                                  <sklearn.compose._column_transformer.make_column_selector object at 0x7f9ebf269080>),
                                                 ('cat_transformer',
                                                  Pipeline(steps=[('cat_imputer',
                                                                   SimpleImputer(strategy='most_frequent')),
                                                                  ('freq_encoder',
                                                                   FrequencyEncoding())]),
                                                  <sklearn.compose._column_transformer.make_column_selector object at 0x7f9ebf2694e0>)])),
                ('regressor', LinearRegression())])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="46b2e558-1ceb-451c-a530-1810bd9ad332" type="checkbox" ><label class="sk-toggleable__label" for="46b2e558-1ceb-451c-a530-1810bd9ad332">preprocessors: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(transformers=[('num_transformer',
                                 Pipeline(steps=[('num_imputer',
                                                  SimpleImputer()),
                                                 ('scaler', MinMaxScaler())]),
                                 <sklearn.compose._column_transformer.make_column_selector object at 0x7f9ebf269080>),
                                ('cat_transformer',
                                 Pipeline(steps=[('cat_imputer',
                                                  SimpleImputer(strategy='most_frequent')),
                                                 ('freq_encoder',
                                                  FrequencyEncoding())]),
                                 <sklearn.compose._column_transformer.make_column_selector object at 0x7f9ebf2694e0>)])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="e04c479c-097f-44f8-a132-5b7116ba0c35" type="checkbox" ><label class="sk-toggleable__label" for="e04c479c-097f-44f8-a132-5b7116ba0c35">num_transformer</label><div class="sk-toggleable__content"><pre><sklearn.compose._column_transformer.make_column_selector object at 0x7f9ebf269080></pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="a75d2ab2-63b5-432e-a8b8-23a45aa9b97d" type="checkbox" ><label class="sk-toggleable__label" for="a75d2ab2-63b5-432e-a8b8-23a45aa9b97d">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="eda15b36-473d-4286-906e-355404dfe7cb" type="checkbox" ><label class="sk-toggleable__label" for="eda15b36-473d-4286-906e-355404dfe7cb">MinMaxScaler</label><div class="sk-toggleable__content"><pre>MinMaxScaler()</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="30d295dc-85d3-4c31-a6b7-0ea3accfcc2a" type="checkbox" ><label class="sk-toggleable__label" for="30d295dc-85d3-4c31-a6b7-0ea3accfcc2a">cat_transformer</label><div class="sk-toggleable__content"><pre><sklearn.compose._column_transformer.make_column_selector object at 0x7f9ebf2694e0></pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="67f85d13-d04a-48ff-8fea-b164f9caec8d" type="checkbox" ><label class="sk-toggleable__label" for="67f85d13-d04a-48ff-8fea-b164f9caec8d">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(strategy='most_frequent')</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="d5fd5490-02cc-4a6c-92f2-61aee83033ee" type="checkbox" ><label class="sk-toggleable__label" for="d5fd5490-02cc-4a6c-92f2-61aee83033ee">FrequencyEncoding</label><div class="sk-toggleable__content"><pre>FrequencyEncoding()</pre></div></div></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="2fc1ceec-4e54-4c70-9c5e-2d419785507a" type="checkbox" ><label class="sk-toggleable__label" for="2fc1ceec-4e54-4c70-9c5e-2d419785507a">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div></div></div>

This pipeline (transformers + estimators) can be used in all kinds of cross-validation or random/grid searches.

## Custom Transformers

Even though scikit-learn provides a lot of transformers, you may find that you want to transform your data in a way that is not implemented in scikit-learn. In this case, you need to create a custom transformer. This can be done by inheriting from the BaseEstimator and TransformerMixin classes. You only need to implement your own `fit()` and `transform()` or `predict()` methods.

Let's say we want to do frequency encoding. Meaning that every category in a categorical column is encoded using the frequency of occurring. Meaning "Category A" occupies 20% of all the values in a categorical variable, the "Category A" should equal 0.2. When we encounter a category, we have not seen during training, we set the frequency of occurrence of this particular category to 0.

```python
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyEncoding(BaseEstimator, TransformerMixin):
  def __init__(self, normalize=True):
    # Initialize frequency dictionary
    self.freq_dict = {}
    self.normalize = normalize

  def fit(self, X, y=None):
    # Check if X is pandas Series or DataFrame
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):

    # Convert to numpy array
      X = X.values

    # 1. Loop over columns
    for i, feature in enumerate(X.T):

      # 2. Calculate (normalized) frequeny
      counts = pd.Series(feature).value_counts(normalize=self.normalize).to_dict()

      # 3. Store counts in dictionary
      self.freq_dict[i] = counts

    return self
  
  def transform(self, X):
    # Check if X is pandas Series or DataFrame
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):

      # Convert to numpy array
      X = X.values

    # 4. Loop over columns
    for i, feature in enumerate(X.T):

      # 5. Convert to series
      series = pd.Series(X.T[i])

      # 6. Check if values in frequency dict
      condition = series.isin(self.freq_dict[i])

      # 7. If present fill with value, otherwise with zero
      encoded_valuess = np.where(condition, series.map(self.freq_dict[i]), 0)

      # 8. Store back into original matrix
      X.T[i] = encoded_valuess

    return X
```

The code above implements a frequency encoder. Because we used the base classes from scikit-learn this transformer can now be used in a pipeline.

<details>
  <summary><b>A more detailed explanation about the implementation of the FrequencyEncoder</b></summary>

  <b>Training</b>:
  <ol>
    <li>Loop over the columns of the table/matrix.</li>
    <li>Count the values/categories in the columns and if we want to normalize them.</li>
    <li>Store values/categories and corresponding (normalized) frequency into a dictionary with the categories as keys and the count as values.</li>
  </ol>

  <b>Transforming</b>:
  <ol>
    <li>Loop over the columns of the table/matrix.</li>
    <li>Convert array to pandas series.</li>
    <li>Check if the categories in the column are in the dictionary (keys).</li>
    <li>If they are, map the values in the directory to the corresponding category (keys) to the values in the column. If the category value is not in the dictionary, fill with a 0.</li>
    <li>Store the encoded values back into the original matrix.</li>
  </ol>

</details>

Let's use the same pipeline we did earlier. However, now we switch out the OneHotEncoder out for our own custom FrequenyEncoder.

```python
# Pipeline of numerical transformers
num_transformer_pipeline = Pipeline(steps=[
  ("num_imputer",  SimpleImputer(strategy="mean")),
  ("scaler", MinMaxScaler(feature_range=(0, 1)))
])

# Pipeline of categorical transformers
cat_transformer_pipeline = Pipeline(steps=[
  ("cat_imputer", SimpleImputer(strategy="most_frequent")),
  ("freq_encoder", FrequencyEncoding(normalize=True)) # Custom transformer
])

# Create preprocessing pipeline
preprocessors = ColumnTransformer(transformers=[
  ("num_transformer", num_transformer_pipeline, make_column_selector(dtype_include="number")),
  ("cat_transformer", cat_transformer_pipeline, make_column_selector(dtype_include="category")),
])

regressor_pipeline = Pipeline(steps=[
  ("preprocessors", preprocessors),
  ("regressor", LinearRegression())
])

regressor_pipeline
```

<style>div.sk-top-container {color: black;background-color: white;}div.sk-toggleable {background-color: white;}label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.2em 0.3em;box-sizing: border-box;text-align: center;}div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}div.sk-estimator {font-family: monospace;background-color: #f0f8ff;margin: 0.25em 0.25em;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;}div.sk-estimator:hover {background-color: #d4ebff;}div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;}div.sk-item {z-index: 1;}div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}div.sk-parallel-item:only-child::after {width: 0;}div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0.2em;box-sizing: border-box;padding-bottom: 0.1em;background-color: white;position: relative;}div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}div.sk-label-container {position: relative;z-index: 2;text-align: center;}div.sk-container {display: inline-block;position: relative;}</style><div class="sk-top-container"><div class="sk-container"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="0aaaf3b3-a2aa-4b8c-bce5-abfd560f9d98" type="checkbox" ><label class="sk-toggleable__label" for="0aaaf3b3-a2aa-4b8c-bce5-abfd560f9d98">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[('preprocessors',
                 ColumnTransformer(transformers=[('num_transformer',
                                                  Pipeline(steps=[('num_imputer',
                                                                   SimpleImputer()),
                                                                  ('scaler',
                                                                   MinMaxScaler())]),
                                                  <sklearn.compose._column_transformer.make_column_selector object at 0x7f9ebf246550>),
                                                 ('cat_transformer',
                                                  Pipeline(steps=[('cat_imputer',
                                                                   SimpleImputer(strategy='most_frequent')),
                                                                  ('freq_encoder',
                                                                   FrequencyEncoding())]),
                                                  <sklearn.compose._column_transformer.make_column_selector object at 0x7f9ebf246438>)])),
                ('regressor', LinearRegression())])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="83adda71-82df-4f23-ae82-059964bc9fc5" type="checkbox" ><label class="sk-toggleable__label" for="83adda71-82df-4f23-ae82-059964bc9fc5">preprocessors: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(transformers=[('num_transformer',
                                 Pipeline(steps=[('num_imputer',
                                                  SimpleImputer()),
                                                 ('scaler', MinMaxScaler())]),
                                 <sklearn.compose._column_transformer.make_column_selector object at 0x7f9ebf246550>),
                                ('cat_transformer',
                                 Pipeline(steps=[('cat_imputer',
                                                  SimpleImputer(strategy='most_frequent')),
                                                 ('freq_encoder',
                                                  FrequencyEncoding())]),
                                 <sklearn.compose._column_transformer.make_column_selector object at 0x7f9ebf246438>)])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="850e239f-c05e-451e-9a92-cd95bdaf4b30" type="checkbox" ><label class="sk-toggleable__label" for="850e239f-c05e-451e-9a92-cd95bdaf4b30">num_transformer</label><div class="sk-toggleable__content"><pre><sklearn.compose._column_transformer.make_column_selector object at 0x7f9ebf246550></pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="19ecaa5d-3401-4eaf-ab44-7defdea82764" type="checkbox" ><label class="sk-toggleable__label" for="19ecaa5d-3401-4eaf-ab44-7defdea82764">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="db97b2af-0dc7-4fd6-a362-da429dab6e6a" type="checkbox" ><label class="sk-toggleable__label" for="db97b2af-0dc7-4fd6-a362-da429dab6e6a">MinMaxScaler</label><div class="sk-toggleable__content"><pre>MinMaxScaler()</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="7dedf6a5-fd39-44ee-b964-8a08801aa6a1" type="checkbox" ><label class="sk-toggleable__label" for="7dedf6a5-fd39-44ee-b964-8a08801aa6a1">cat_transformer</label><div class="sk-toggleable__content"><pre><sklearn.compose._column_transformer.make_column_selector object at 0x7f9ebf246438></pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="cd18fe76-26f3-4ec2-a646-7434a4c40944" type="checkbox" ><label class="sk-toggleable__label" for="cd18fe76-26f3-4ec2-a646-7434a4c40944">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(strategy='most_frequent')</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="94f6caef-53b3-45a8-8401-1c2364070a89" type="checkbox" ><label class="sk-toggleable__label" for="94f6caef-53b3-45a8-8401-1c2364070a89">FrequencyEncoding</label><div class="sk-toggleable__content"><pre>FrequencyEncoding()</pre></div></div></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="1b74220c-0f5d-4965-abdd-41e55a6e920e" type="checkbox" ><label class="sk-toggleable__label" for="1b74220c-0f5d-4965-abdd-41e55a6e920e">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div></div></div>

We can now fit the entire pipeline containing our custom transformer on the dataset. Furthermore, we can use the `score()` method or perform cross-validation. We can also persist the entire pipeline to make use of it later on in production. We only need to load our persisted model pipeline, and call the `predict()` method on new data.

```python
regressor_pipeline.fit(X_train, y_train)
score = regressor_pipeline.score(X_test, y_test)

print(f"R-squared: {score:.2f}")
```

```shell
R-squared: 0.81
```

## Conclusion

In this post, I wanted the demonstrate how powerful and useful the scikit-learn's Pipeline API is. We can easily implement any data transformation or model estimator leveraging some scikit-learn base functionality. We can then persist the whole data transforming and modeling pipeline into a single serialized artifact, which is great when we want to use this model later on.

The pipeline demonstrated in this post is really simple, and only scratches the surface of what is possible when using them.