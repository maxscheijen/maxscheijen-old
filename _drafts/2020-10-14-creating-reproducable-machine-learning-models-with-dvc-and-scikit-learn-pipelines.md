---
layout: post
title: "Reproducible Machine Learning is challenging"
categories: [data-science, machine-learning, reproducability, deployment]
author: Max Scheijen
---

In my view, machine learning models can be separated into two different categories. Machine learning models that are reproducible and models that are not. If machine learning models are not reproducible, how do we know that they work? We should assume they are not working, or least we should ask the question when can we say with some certainty that they do.

Often reproducibility is only seen as necessary in scientific or academic work. However, this is not the case. Organizations or businesses that deploy machine learning models also need to be able to reproduce models and corresponding results, which I will demonstrate later on in this article. This will highlight the importance of reproducibility in machine learning and why it is more challenging than you initially may think.

## Defining reproducibility

Before discussing reproducibility in the machine learning context, we need to define the concept. Reproducibility refers to the re-doing of computations or the re-doing of experiments {% cite fidler_2018 %}. Being able to reproduce experiments is one of the pillars on which the scientific method is built. When building machine learning models, we are doing a lot of experiments. But what does reproducibility mean in the context of machine learning or data science?

> **Reproducibility** in machine learning is the ability to exactly duplicate a model in a later stage, given the same data as input. 

We want the ability to go back to an earlier version of our machine learning system. Then retrain our model, which then produces the same output predictions as we first trained the model, on the same test data {% cite galli_2020 %}.

Easy right? It is not. In real-world machine learning applications rolling back to two previous models can be quite hard. In this article, I discuss why reproducibility is necessary for machine learning systems, both from a business and regulatory standpoint, and what makes it reproducing results difficult.

## Why is reproducibility important

But why should you care? As I stated in the introduction, if your model can not be reproduced, we can not verify if it works. This should be enough. However, there are additional reasons why you should care. The lack of reproducible machine models is a problem that any machine learning practitioner will encounter, whether in industry or academia {% cite sugimura_2018 %}. In academia, it is necessary to verify your research findings. However, this is not my main focus. I focus on arguments in favor of reproducibility from a business perspective.  

Lack of reproducibility can have numerous negative impacts on an organization. For example, let us say our data scientists create a well-performing predictive model in their research environment. But when we want to put this model into production, we can not reproduce the results of the research environment. A model that can not recreate the results into production does not add any value to our business. This results in wasted time and effort, in addition to the potential **financial loss**. The machine learning model serves little use outside of the production environment. We never know if we have developed a better performing model. We can not accurately assess if the new model performs better than the previous one. You should not deploy a new model into production if you can not verify if it performs better {% cite galli_2020 %}.

Cost is not the only reason why you should strive to create reproducible machine learning models. There is also a **regulatory** need for reproducibility. Some industries are under scrutiny by regulatory authorities, especially after the introduction of the General Data Protection Regulation in the European Union. Regulatory authorities can request how a prediction by a machine learning model was made {% cite samiullah_2019 %}. However, this prediction was made in an earlier version of the model. The current model makes other predictions given the used data. We need to go back to previous versions. Therefore we need to think carefully about designing our machine learning model pipeline. We need to keep in mind that we need to be able to reproduce earlier models and data {% cite samiullah_2019 %}.

## Why reproducibility in Machine Learning systems is difficult

So why do I keep saying that creating reproducible machine learning models is difficult? In contrast to traditional software engineering, where we build deterministic software, machine learning systems are often stochastic. Meaning there is randomness involved in constructing these systems. 

Where traditional deterministic software can more easily be version controlled, this challenging when dealing with the randomness of stochastic machine learning systems. When software is deterministic, it is possible to easily roll back to earlier versions of the system. However, machine learning systems are often stochastic. They rely on some sort of randomness for their build. This makes them much harder to version because randomness is hard to reproduce.  

However, the stochastic nature of machine learning algorithms is not the only difficulty. Machine learning systems consist of both code and data, and both of them can change. Changes in one of these components result in a different model {% cite gorcenski_2019 %}. Therefore we need to also versional control the data because our machine learning model is built/trained on the data. If the data changes, the machine learning model changes, even when we find a way to control the stochastic nature of some of the models.

To fully reproduce machine learning systems, we need to create a completely reproducible machine learning pipeline. Sugimura and Hart (2018) present some components to consider when creating reproducible models {% cite sugimura_2018 %}. Let's take a closure look at the steps involved in creating a reproducible machine learning pipeline.

1. The minimum requirement is to we should save and, **version control the code** which processes the data and trains the model.
2. We should specify the **version of all the software and packages** used in creating the pipeline.
3. We must record how we obtain **data**, which specific samples, and how we process them.
4. If we use some sort of random or grid search to find **hyperparameters**, we need to save them.
5. It can also be a good idea to record the hardware used to train the model.

Basically, we need a way to version the entire machine learning pipeline {% cite sugimura_2018 %}. Therefore we need to build for reproducibility from the start.

Samiullah states in a blog post about productizing machine learning models: "Persist all model inputs and outputs, as well as all relevant metadata such as config, dependencies, geography, timezones and anything else you think you might need if you ever had to explain a prediction from the past. Pay attention to versioning, including your training data" {% cite samiullah_2019 %}.

<!-- ![png]({{ site.url }}/assets/img/2020-10-14-reproducable-ml-data-model-code.svg) -->

So, creating reproducible machine learning systems is challenging because it relies on data to build, in addition to the often stochastic nature of these systems. If we want to create reproducible machine learning systems from the start, what are the most important parts we need to take into account? Let's take a closer look at the main components of a machine learning system: code, data, and models.


## Reproducible components of Machine Learning Systems

Before discussing the main components of machine learning systems, I need to highlight the importance of **software versions**. Controlling the software used to create machine learning systems may be the easiest part of the pipeline, especially when using open-source software. Always record the version of the software or software package you use in the most detailed manner. For example, record the python version you used (3.7.2) and the version of scikit-learn you used to create the model. Sometimes default parameters change between versions. This can result in a possible different machine learning model and output on the same data.

### Code

Let's first discuss the code component of our pipeline. This is the most likely the most intuitive way to create a reproducible system if you are familiar with a more traditional software engineering approaches. We can version-control systems for tracking changes in our source code. Git is probably the most used and well known version-control software. We can connect our Git repositories with GitHub or Gitlab for additional functionality and easier collaboration.

### Data

The data component of the pipeline becomes more challenging. We need to keep a record of the data we used to train our machine learning model. If the dataset on which the model has trained changes creating the same model at a later stage is often not possible. We need a way to keep track of all the changes in the dataset, so we use the same dataset containing the same information to recreate or retrain our model. A perfect solution has yet to become available. However, I think at this moment, Data Version Control (DVC) has the most potential. DVC is basically git for data and model files. It allows to version control your datasets.


### Model 

In addition to the data, we also need a way to version the model. Often machine learning models require randomness for training. Basically, we need a way to record how the model was trained. This randomness can cause slight differences when we want to reproduce a model. Even though predictions would not vary much, it is advised to some random seed if possible. This ensures a better reproducible model. Also, keep track of your hyper-parameters, these can influence how your model behaves. However but also keep track of transformations we made to the data. Often transformations are learned. For example, the mean and the standard deviation used normalization or scale the data depends on the underlying data. If we use ensemble methods, we need to record the structure of the ensemble. Often this can be done by versioning the code.

<!-- In addition to the data, we also need a way to version the model. Basically, we need a way to record how the model was trained. This not only means that we record the hyper-parameters of the model, but also keep track of transformations we made to the data. For example, the mean and the standard deviation used normalization or scale the data depends on the underlying data.

- Model provenance refers to the record of how a model was trained. This includes the order of the features, the applied feature transformations (e.g. standardization), the hyperparameters of the algorithm, and the trained model itself. If the model is an ensemble of submodels, then the structure of the ensemble must be saved (Sugimura & Hartl). 

- The machine learning algorithms themselves also cause significant challenges to reproducibility. Similarly to some instances of feature creation, certain machine learning models require randomness for training. Common examples of this scenario include tree ensembles, cross validation, and neural networks. Tree ensembles require random feature and data extraction, cross validation relies on random data partitions, and neural networks use randomness to initialize their weights. The randomness causes slight differences between models, even ones with the same training data; these models then won’t meet the requirements of reproducibility (Soledad Galli).

- Another potential problem may arise when working with arrays. Certain APIs used to build models utilize arrays rather than data frames. Unlike data frames, arrays don’t have named features, so ordering of the columns is the only way to reliably identify them. In these cases, programmers and data scientists will need to devote additional attention to ensure to always pass the features in the correct order (Soledad Galli).

- Similar to feature creation, simple solutions can combat most of these threats to reproducibility. Data scientists must give extra care to record orders in which they pass features, the hyper-parameters used, set the seeds when needed, and mind the structure if the final model is an ensemble of models (Soledad Galli).


# Creating reproducable pipelines in Python

As you shift from the Jupyter notebooks of the research environment to production- ready applications, a key area to consider is creating reproducible pipelines for your models (Christopher Samiullah).

- Gathering data sources.
- Data pre-processing
- Variable selection
- Model building


- DVC: Open-source Version Control System for Machine Learning Projects. DVC tracks ML models and data sets DVC is built to make ML models shareable and reproducible. It is designed to handle large files, data sets, machine learning models, and metrics as well as code.

- Sklearn: is an industry standard Python machine learning library with many extremely useful and powerful modules. Modularity, or splitting up the project into components, will help ensure generality and scalability. For compatibility, the architecture will follow Scikit-learn’s API conventions, which are considered the industry standards. Alone, these packages aren’t sufficient for reproducibility as they are limited to a single training model.
  

Whilst it is possible to write custom code to do this (and in complex cases, you may have no choice), where possible try and avoid re-inventing the wheel. Reproducing containerized systems is much easier because the container images ensure operating system and runtime dependencies stay fixed. The ability to consistently and quickly generate precise environments is a huge advantage for reproducibility during testing and training (Christopher Samiullah).

```shell
.
├── Makefile # make file for automation
├── Pipfile # virtual environment
├── Pipfile.lock
├── data # dataset directory
│   └── train.csv # train data file
├── requirements.txt # package requirements
├── setup.py # create package
├── src # directory containing main source files
│   ├── config.py # global configs
│   ├── dataset.py # create dataset
│   ├── features.py # cretea features
│   ├── model.py # model architecture
│   └── train.py # train model
└── tests # directory containing tests
    ├── conftest.py 
    ├── test_dataset.py
    ├── test_features.py
    ├── test_model.py
    └── test_train.py
└── trained_model
    └── model.pkl
```



- Model and Data versioning

## Software
Managing the software used to create our machine learning models is probably the easiest step to reproduce. However, it has to be done with care. The easiest way to create an isolated environment in which we train machine learning model. In this isolated environment we install our python and third party packages. Make sure to specify versions both the python and the used package versions. For creating an isolated environment I recommend either using [`venv`](https://docs.python.org/3/library/venv.html), [`virtualenv`](https://virtualenv.pypa.io/en/latest/) or [`pipenv`](https://pipenv-fork.readthedocs.io/en/latest/) to install python and python packages. You could also use docker, however I want to keep this article as simple as possible. 

The `requirements.txt` file below contains the python package requirements for our machine learning application. Note that it's important to specfiy the exact version of the packages. Often default parameters change between package versions. In this example I will use `Pipenv` to install the requirements. Is package also allows us the specify the python version.
```shell
# requirements.txt
pandas==1.1.2
scikit-learn==0.23.2
dvc==1.8.1
```

Let's now use `pipenv` as our package manager and virtual environment. First we install `pipenv` using pip. After this we use `pipenv` to install python 3.7 and the packages stated in our `requirements.txt` file. 

```shell
# install pipenv
$ pip install pipenv

# install python
$ pipenv --python 3.7

# install packages
$ pipenv install -r requirements.txt

# activate environment shell
$ pipenv shell
```

We now have an isolated python 3.7 environment in which we can install packages and build a reproducible machine learning model.

## Code

The machine learning system or model architecture is expressed in code. Therefore is quite easy to reproduce across different version or iterations when using version control. It follows the same version control process as more tradition software systems. I suggest using [`git`](https://git-scm.com/) as it is the most used version control system and integrates nicely with both GitHub and GitLab, which provide addition functionality. Version control is the minimum we need to create a reproducible code. However there are some code best practices which are recommended, to make your code more accessible and readable for others. 

```

```

I recommend to make a global config file, which contains all your global parameters.  These configs can be accessed across you machine learning systems code. I urge to set a random seed, which you should apply to every random process which your machine learning model pipeline (data splitting, stochastic modelling, etc.). This can contain the path to your training data, train model directory. However it can also for example contain model hyper paramaters, splitting sizes.


```python
# config.py
from pathlib import Path

class Config:
  ROOT = Path(__file__).parent
  DATA_DIR = ROOT / "data"
  TRAIN_DATA = DATA_DIR / "train.csv"
  MODEL_DIR = ROOT / "trained_model"
```

The model directory should  only contain the trained model binary. This prevents confusion on what is the output of the training procedure. 

## Data

Reproducing data is on of the more challenging parts in creating reproducible machine learning models. However, is earlier stated it is essential because the our model depends on the data as input to learning its parameters. Different data will lead to different model parameters and therefore will most likely have a different output on test data. 

```python
# data.py
```

## Model

```python
# model.py
```

After running the training script we are left with an artefact, the model binary. This model can be served and used to make predictions. However, versioning this model is not always necessarily. Because we've build a reproducible machine learning pipeline, re-running the training script will reproduce the exact model binary. However if computation is long or expense we can chose the version the trained model binary in the same way was we version our data.
 -->

## References

{% bibliography --cited %}
 
