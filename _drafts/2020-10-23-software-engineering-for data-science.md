---
layout: post
title: "Software Engineering for Data Science"
categories: [data-science, machine-learning, software-engineering]
author: Max Scheijen
---

Data science and machine learning are relatively new fields, overlapping with both statistics and computer science. Most practicing Data scientist do not have a software engineering background. Often them come from statistics or other fields. However what can we as Data Scientist learning from computer science and software engineering.

Most people practicing data science follow a somewhat similar workflow. We iteratively develop our models by gathering data, cleaning data, creating features, and training machine learning models. However, this process is far from linear. Often we need to go back some steps in the process, do some more cleaning or feature engineering, and re-train our model again.

This articles address the most import parts of writing good production ready code. First I aruge why you should note use jupyter notebooks for sharing or production. After this, I go over some good practices that will improve you applications such as: clean code, modularity, refactoring, code efficieny, code versioning, testing and logging. Because most data science and machine learning systems and applications are written in Python, this article will discuss best practices for this progamming language. However, these guides can be applied to any langauge.

## Notebooks

So let us first talk about the Jupyter notebook. Data Scientists frequently use Jupyter notebooks to do analysis and model training. Jupyter notebooks have become an essential tool in the Data Sciencists and the Machine learning practioners toolkit.

> The [Jupyter Notebook](https://jupyter.org/) is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text. Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, machine learning, and much more.

The mixing of markdown, code and plots leads to well-documented code. This makes it easy to communicate and report to your results with non-technical members of your organization. Furthermore they are great for experimenting. We can create different machine learning models and get almost instand feedback on them. or hyperparameter tuning. However, Jupyter notebooks have some disadvantages.

### Problems with notebooks

They do not encourage you to write good code or to follow best coding practices. This is fine if you are the only one using our code, and your code never makes it into production. However, when working with others or your code needs to run stable in production, this becomes different and hard using notebooks. Furthermore, reproducing data processing steps, model training, and deploying the model becomes quite tricky and burdensome when using notebooks.

Say we have several classes and functions that help us transform data. If we use notebooks, we are almost forced to store them all in one place, the notebook. If we need to change some function, we need to run all the code again. Because we do not know the [current state of the code cells](https://www.youtube.com/watch?v=7jiPeIFXb6U). Because Jupyter notebook cells can be run out of order, they have lots of unobservable states, which often can not be known. That makes screw ups much more likely to happen. If you share your code with someone else, after you have run the code cells out of order, this person will not be able to recreate your results.

### Versioning notebooks

Furthermore, notebooks are difficult to [version](https://owainkenwayucl.github.io/2017/10/03/WhyIDontLikeNotebooks.html). Jupyter notebooks containing Python code are not Python files. They are basically large JSON objects. If we commit the notebook after changes, the diff becomes really big, making it difficult to review them or to merge into main branch. This makes it challenging to use them in teams. If we want to run machine learning models into production using larger systems, we need to be ablet to test our code base. Test if our code behaves as we expect it to. Eventough testing can be done, it is still difficult.

### Learning from Software Engineering

As traditional software engineering is much more established and has more best encoded and enforced best practices, what can we learn from them? What are their best practices, and can they be applied to the machine learning and data science field? This article tries to answer this question.

Do not get me wrong, I like working with a notebook. They are greate for beginners to experiment with data science and machine learning, by providing instant feedback. However, as soon as you need to share your code, or deploy a machine learning system into production, notebooks become quite challenging to work with. This defeats the purpose of these notebooks. After the first experimental steps, notebooks are useful when combining code and markdown to create insights that can be reported.

There are tools trying to address the problems with notebooks. Such as [nbdev](https://github.com/fastai/nbdev) that lets you create full Python projects using notebooks, or [papermill](https://papermill.readthedocs.io/) that allow for execution of notebooks in a production ecostem. These tools are not widely used in production, and if they on a later date will, it is always good to have and apply good software engineering.

## Clean code

You should always write clean code. This not only helps your teammates better understand your code, but also helps you understand you own code when to come back to it a year later. Generally you should follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) conventions. PEP 8 documents the guidelines, and best practices on how to write Python code, the goal to improve readability, and consistency in Python code. However, often Data scientists (including myself) do not follow or at least do not follow these conventions enough. Let me highlight some essentials, which I think you should always follow and that will help you write cleaner code.

### Nameing styles

Variables names should follow always follow the same [style](https://realpython.com/python-pep8/#naming-styles). A constant should always be upper case. Classes should start with an uppercase letter and use camel case for separate words `class MeanImputer:`. Modules, methods and functions should be lowercase, short, and separated by an underscore.

Below displays the nameing style of constants. In machine learning we often use constants inside a config file. Using uppercase charachters makes it clear to to yourself and your team which parameters to change.

```python
BASE_DIR = pathlib.Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SPLIT_SIZE = 0.7
N_ESTIMATORS = 100
```

Below displays the nameing style of a class. Often we need to write data processing or dataset creating classes.

```python
from sklearn.base import TransformerMixin


class MeanImputer(TransformerMixin):
    pass
```

Below displays the nameing style of a function. The words in the names should be seperated by a lowercase.

```python
import pandas as pd


def load_data(path):
    return pd.read_csv(path)
```

### Nameing variables appropriately

You should really think about how to [name your variables](https://realpython.com/python-pep8/#naming-styles). I can almost guarantee that you encountered a case where a csv is loaded into a data frame with the name `df`. The variable name `df` is not descriptive of what the content of it represents. `cars` would be a better variable name. This will give you and your teammates insight into what the variable contains.

```python
import pandas as pd


# Not informative
df = pd.read_csv("cars.csv")

# More informative
cars = pd.read_csv("cars.csv")
```

### Blank lines

As we all know b[lank lines](https://realpython.com/python-pep8/#blank-lines) can improve the readability of our code. It keeps your code organized and less overwhelming for yourself and others. Top-level functions and classes should sperated by two blank lines. Methods inside an object are surrounded by a single blank line. Sometimes it is helpful if a function is complicated to separate different steps inside the function with blank lines.

### Comments

Your Pyhton scripts should contain a lot of [comments](https://realpython.com/python-pep8/#comments). Comments help you document your code, and collaborators to understand what your code is doing. You should use full sentences. Think carefully about what to comment. At a minimum you should comment the complex logic of your code.

```python
mean_dict = {} # Initialize dictionary

for feature in X.columns:
    # Loop over dataframe columns and store mean in dictionary
    mean_dict[feature] = X[feature].mean()
```

You can also use inline comments, however use the sparingly. Do not use them to explain the obvious as I demonstrated in the code block above.

### Docstrings

We can use [docstrings](https://realpython.com/python-pep8/#documentation-strings) to explain and document a specific block of code. You should write them for all modules, functions, classes and methods. A nicely written docstring can also use by tools like Sphinx to automaticlly build nicely looking documentation.

```python
from sklearn.base import TransformerMixin


class MeanImputer(TransformerMixin):
    def __init__(self):
        self.mean_dict = {}

    def fit(self, X: pd.DataFrame, y: None) -> "MeanImputer":
        """Learns the mean values

        Parameters
        ----------
        X : pd.DataFrame
            The training input samples.
        y : None or pd.Series
            Not needed for imputation.
        """
        for feature in X.columns:
            self.mean_dict[feature] = X[feature].mean()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms the variables by replacing missing values with the mean

        Parameters
        ----------
        X : pd.DataFrame
            The DataFrame to transform

        Returns
        -------
        pd.DataFrame
            The transformed and imputed DataFrame
        """
        X = X.copy()

        for feature in X.colums:
            X[feature] = X[feature].replace(np.nan, self.mean_dict[feature])

        return X
```

The code above shows the implementation of a custom mean imputer by inhereting from the a sklearn base transformer class. The functionality of the class is not important but it demonstrates the use of a docstring. The docstring is written in the numpy style.

### Type hinting

In Python [type hinting](https://docs.python.org/3/library/typing.html) is a way to statically indicate the type of a value in code. Eventhough Python does not enforce function and variable annotations, they can be used by third party tools suchs as type checkers, IDEs, and linters.

```python
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load CSV data into a DataFrame

    Parameters
    ----------
    path : str
        path to CSV

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame containing values of the CSV
    """
    return pd.read_csv(path)
```

In the code above we can see a function with type hints. We indicate that the functions takes an argument which should be a string. The function then needs to return a pandas DataFrame.

## Code modularity

Modularity is one of the most crucial concepts in creating robust applications. We can use the [Do not Repeat Yourself (DRY)](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself) principle. You should generalize and consolidate your code as much as possible. Functions for example should only have one job, abstracting your logic without over-engineering. However, you should keep an eye out for creating too many modules. Use your judgment, and if you are inexperienced, have a look at popular GitHub repositories such as scikit-learn and check out their coding style.

- CookieCutter Data Science

## Refactoring

Refactoring is a technique for improving the design of an existing code base by changing the applicatio#n in such a way that it behaves the same way, but the internal structure is improved. These improvements can increase stability, performance, and reduce complexity.

## Code efficiency

## Code versioning

Version control is a way to manage changes in software. One of the most used version control systems is Git. This version control software integrates nicely with platforms like GitHub or GitLab.

## Testing

Testing can be very usefull for Data Science and Machine Learning systems. If we do not test, we have no way of telling wether or systems functions the way we want.

## Logging

Logging can help us to better understand the flow of our machine learning application. It can also helps us to discover bugs, or log the results of our predictions in production.

## Conclusion

In this article, I discussed some software engineering principles which Data Scientist or Machine Learnin practionores can used to write better prediction systems.