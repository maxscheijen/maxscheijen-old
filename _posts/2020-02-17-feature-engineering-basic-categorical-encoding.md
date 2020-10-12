---
layout: post
title: "Feature Engineering: Categorical Data"
categories: [data-science, machine-learning, feature-engineering]
author: Max Scheijen
---

Good and robust feature engineering can improve many machine learning models. One way to create new features is by encoding categorical variables. Categorical features contain qualitative information. Many machine learning algorithms can only deal with numerical data. However, these categorical variables are often encoded as strings. There are many techniques to encode these features. In this post, we look at several basic methods of transforming these variables to some numeric representation. I highlight the pros and cons of these encoding techniques for the use of some of the commonly used machine learning models.

We'll use some artificial data to demonstrate these techniques. I also use term variables and features interchangeably.

## Categorical Data types

There are two main types of categorical data. Nominal data has <u>no intrinsic order</u> to it. Examples of these features are the cities, street names, gender, and many more. Ordinal features are different because there is an <u>intrinsic order</u> to the categories. Often surveys use the <a href="https://en.wikipedia.org/wiki/Likert_scale" target="_black">Likert scale</a> to get opinions about subjects. This scale as an order to it and ranges from strongly disagree to strongly agree. The data type of the variable influence the technique we use to encoding that particular feature.

## One-Hot Encoding

One-hot encoding or dummy encoding is a technique where we encode a categorical by indicating if the category is present for each observation.  This technique is most used on nominal categorical data. Generally, we can encode the different labels by $k$-1 binary features. Where $k$ is the number of categories present in the variable. For example, the variable biological sex has two different variables (female and male) and therefore has a $k$ value of 2.

```python
# load data processing library
import pandas as pd

# load and display data
sex = pd.read_csv('sex.csv')

# create copy of data
sex_ohe_1 = sex.copy()

# use get_dummies and drop_first=True for k-1 one-hot encoding
sex_ohe_1['male'] = pd.get_dummies(sex_ohe_1, columns=['sex'], drop_first=True)

# display encoding
sex_ohe_1
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
      <th>sex</th>
      <th>male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>female</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>female</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>female</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>female</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

The advantage of one-hot encoding with $k$-1 features is that we can represent the whole dataset with one less dimension. However, we can also use $k$ binary features to represent a categorical variable. If we want to asses the importance of every single feature it is better to use as my groups as labels. Often tree-based models perform better on data that is encoded in $k$ features. If we use the same example as above the encoding look like the following:

```python
# create copy of data
sex_ohe_2 = sex.copy()

# use get_dummies and drop_first=False for k one-hot encoding
sex_ohe_2[['female', 'male']] = pd.get_dummies(sex_ohe_2, columns=['sex'])

# display encoding
sex_ohe_2
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
      <th>sex</th>
      <th>female</th>
      <th>male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>male</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>female</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>male</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>male</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>female</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>female</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>female</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

The main advantages of one hot encoding are that it doesn't assume a particular distribution of the data, keeps all the information and can also be used with linear models. However, this technique can expand the feature space a lot if there are many different labels in the categorical variable (high cardinality).

In python, you can also use `OneHotEncoder` method from `sklearn` to implement one-hot encoding.

<div class="alert alert-warning">
  <strong>Note: </strong> Normally you would note keep the original feature, this is purely for demonstration.
</div>

## Label Encoding

Label encoding is one of the most used methods to encode ordinal categorical features. However, it can also be used with a nominal variable. We replace the labels in a feature with integer starting at 0 to $n$. Where $n$ is the number of labels in the feature. 

```python
# load LabelEncoder
from sklearn.preprocessing import LabelEncoder

# load cars data
cars = pd.read_csv('cars.csv')

# create copy of data
cars_le = cars[['car_brand']].copy()

# apply labelencododer to car_brand feature
cars_le['car_encoded'] = cars_le[['car_brand']].apply(LabelEncoder().fit_transform)

# display encoding
cars_le
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
      <th>car_brand</th>
      <th>car_encoded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ford</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>toyota</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>toyota</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ford</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ford</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>toyota</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ford</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>bmw</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ford</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>toyota</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>

One advantage of label encoding over one-hot encoding is that this method does not expand the feature space. We do not create any new features using this technique.

Even though label encoding also works well with tree-based models this technique can lead to some problems when used with linear models. If we use label encoding on nominal variable linear models assume that there is some order to the encoded variable even when there is none.  Tree-based models don't have this problem.

One-hot and label encoding are probably the most used categorical encoding techniques.

## Frequency Encoding

When using frequency encoding we replace the label in a categorical feature with the percentage of that particular label in the variable. This method assumes that the number of observations shown by each category is predictive of our target. Frequency encoding is often used in data science competitions.

```python
# create copy of data
cars_fe = cars[['car_brand']].copy()

# get frequency of the car_brands
counts = cars_fe['car_brand'].value_counts().to_dict()

# apply frequency and divide by number of observations for frequency
cars_fe['freq_enc'] = cars_fe['car_brand'].map(counts) / len(cars_fe)

# apply frequency for count
cars_fe['count_enc'] = cars_fe['car_brand'].map(counts)

# display encoding
cars_fe
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
      <th>car_brand</th>
      <th>freq_enc</th>
      <th>count_enc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ford</td>
      <td>0.5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>toyota</td>
      <td>0.4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>toyota</td>
      <td>0.4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ford</td>
      <td>0.5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ford</td>
      <td>0.5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>toyota</td>
      <td>0.4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ford</td>
      <td>0.5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>bmw</td>
      <td>0.1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ford</td>
      <td>0.5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>toyota</td>
      <td>0.4</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>

When we use frequency or count as the same pros and cons as the label encoding method. It is also possible that different labels get the same encoding. If two categories occur the same number of times.

However, it can be a great way of encoding categorical features with high cardinality.  It assumes that there is a connection between the frequency of the label and the target,  <a href="https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02" target="_black"> "it helps the model to understand and assign the weight in direct and inverse proportion, depending on the nature of the data"</a>.

## Target Encoding

The target encoding technique uses the mean of the target feature to replace the categorical variable. By doing so we take into account the number of labels with the target feature. This way we can decrease cardinally in the variable.

```python
# create copy of data
cars_te = cars.copy()

# groupby categorical_variable and get mean of target variable
target = cars_te.groupby('car_brand')['target'].mean()

# map target mean variable for category
cars_te['target_enc'] = cars_te['car_brand'].map(target)

# display encoding
cars_te
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
      <th>car_brand</th>
      <th>target</th>
      <th>target_enc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ford</td>
      <td>1</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>toyota</td>
      <td>1</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>toyota</td>
      <td>0</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ford</td>
      <td>0</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ford</td>
      <td>0</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>toyota</td>
      <td>0</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ford</td>
      <td>1</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>bmw</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ford</td>
      <td>1</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>9</th>
      <td>toyota</td>
      <td>1</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>

The main con of this technique is its dependency on the distribution of the target. This can lead to over-fitting. You can use <a href="https://en.wikipedia.org/wiki/Additive_smoothing" target="_blank">additive smoothing</a> to counter this. It can also lead to encoding several different categories with the same numerical encoding.

## Conclusion

In this post, we looked at some basic feature engineering by highlighting several techniques to encode categorical data into a numerical representation. There are many more methods to encode categorical variables. You can experiment with these techniques to see which makes the best predictive machine learning model.

## References

* <a href="https://www.udemy.com/course/feature-engineering-for-machine-learning/" target="_blank">Feature Engineering for Machine Learning</a> by Soledad Galli
* <a href="https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02" target="_blank">All about Categorical Variable Encoding</a> by Baijayanta Roy
