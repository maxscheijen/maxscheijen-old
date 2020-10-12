---
layout: post
title: "Reproducible Machine Learning"
categories: [data-science, machine-learning, reproducability, deployment]
author: Max Scheijen
---

In my view, machine learning models can be separated into two different categories. Machine learning models that are reproducible and model that are not. If machine learning models are not reproducible, are they actually working? They are not, or least when can with some certainty say they do.

Often reproducibility is only seen as necessary in scientific or academic work. However, this is not the case. Organizations or businesses that deploy machine learning models also need to be able to reproduce models and corresponding results, which I will demonstrate later on in this article. This will highlight the importance of reproducibility in machine learning and why it is more challenging than you initially may think.

## Defining reproducibility

Before discussing reproducibility in the machine learning context, we need to define the concept. Reproducibility refers to the re-doing of computations or the re-doing of experiments {% cite fidler_2018 %}. Being able to reproduce experiments is one of the pillars on which the scientific method is built. When building machine learning models, we are basically doing a lot of experiments. But what does reproducibility mean in the context of machine learning or data science?

> I define reproducibility in machine learning as the **ability to exactly duplicate a model in a later stage, given the same data as input**. We want the ability to go back to an earlier version of our machine learning system. Then retrain our model, which then produces the exact same output predictions as we first trained the model, on the same test data {% cite galli_2020 %}.

Easy right? It is not. In the real-world machine learning applications rolling back to two previous models can be quite hard. In this article, I discuss why reproducibility is necessary for machine learning systems, both from a business and regulatory standpoint and what makes it reproducing results difficult.

## Why is reproducibility important

But why should you care? As I stated in the introduction, if your model can not be reproduced, we can not verify if it works. This should be enough. However, there are additional reasons why you should care. The lack of reproducible of machine models is a problem that any machine learning practitioner will encounter, whether in industry or academia {% cite sugimura_2018 %}. In academia, it is necessary to verify your research findings. However, this is not my main focus. I focus on arguments in favour of reproducibility from a business perspective.  

Lack of reproducibility can have numerous negatively impact an organization. For example, let us say we our data scientists create a well-performing predictive model in their research environment. But when we want to put this model into production, we can not reproduce the results of the research environment. A model that can not recreate the results into production does not add any value to our business. This results in wasted time and effort, in addition to the potential **financial loss**. The machine learning model serves little use outside of the production environment. We never know if we have developed a better performing model. We can not accurately assess if the new model performs better than the previous one. You should not deploy a new model into production if you can not verify if it performs better {% cite galli_2020 %}.

Cost or not the only reasons why you should strive to create reproducible machine learning models. There is also a **regulatory** need for reproducibility. Some industries are under scrutiny by regulatory authorities, especially after the introduction of the General Data Protection Regulation in the European Union. Regulatory authorities can request how a prediction by a machine learning model was made {% cite samiullah_2019 %}. However, this prediction was made an earlier version of the model. The current model makes other predictions given the used data. We need to go back to previous versions. Therefore we need to think carefully about designing our machine learning model pipeline. We need to keep in mind that we need to be able to reproduce earlier models and data {% cite samiullah_2019 %}.

## Reproducability in Machine Learning systems is difficult

So why do I keep saying that create reproducible machine learning models is difficult? In contrast to traditional software engineering, where we build deterministic software. There is no randomness involved in constructing these systems. Therefore they can easily be versioned using version control. This makes it possible to easily roll back to earlier versions of the system. However, machine learning systems are often stochastic, they rely on some sort of randomness for their build. This makes them much harder to version.  Machine learning systems consist of both code and data, and both of them can change. Changes in one of these components result in a different model {% cite gorcenski_2019 %}.

To fully reproduce machine learning systems we have to take several steps:

1. The minimum requirement is to we should save and **version control the code** which processes the data and trains the model. 
2. We should specify the **version of all the software and packages** used in creating the pipeline.
3. We must record how we obtain **data**, which specific samples, and how we process them.
4. If we use some sort of random or grid search to find **hyperparameters**, we need to save them. 
5. It can also be a good idea to record the hardware used to train the model.

Basically, we need a way to version the entire machine learning pipeline {% cite sugimura_2018 %}. Therefore we need to build for reproducibility from the start.

> Persist all model inputs and outputs, as well as all relevant metadata such as config, dependencies, geography, timezones and anything else you think you might need if you ever had to explain a prediction from the past. Pay attention to versioning, including of your training data {% cite samiullah_2019 %}.

Let's take a closer look at some of these components. 

<!-- ### Software

For full reproducibility, the software versions should match exactly. Save the versions of every software package in the environment.

### Code
- Of all the components of a machine learning system, code is probably the best understood among technologists, because we have been working on building deterministic software systems for a while, and continuous delivery principles are pretty well understood in this context (Emily F. Gorcenski). 
- Feature provenance refers to the historical record of how a feature is generated. Any change to how a feature is generated should be tracked and version controlled (Sugimura & Hartl).

### Data

Data provenance refers to the historical record of how the data of interest was collected. We have found that this is the most difficult challenge to ensure full reproducibility. If the dataset used to train a model changes after the time of training, then it may be difficult or impossible to reproduce a model. This usually occurs in two different ways. The first is when part of the training dataset is deleted or made unavailable. The second is more subtle, and occurs if the dataset is updated (Sugimura & Hartl).

- Since the machine learning model is highly dependent on the data used to train it, data gathering is one of the most significant and difficult challenges to address when it comes to reproducibility. A model will never come out exactly the same unless the exact same data and processes are used to train it (Soledad Galli).

- This approach is used by tools like dvc. Simply put, we emit our data to some storage solution, hash the data in a meaningful way, and write a small stub that we can commit to source control alongside our code (Emily F. Gorcenski).

## Model 
- Model provenance refers to the record of how a model was trained. This includes the order of the features, the applied feature transformations (e.g. standardization), the hyperparameters of the algorithm, and the trained model itself. If the model is an ensemble of submodels, then the structure of the ensemble must be saved (Sugimura & Hartl). 

- The machine learning algorithms themselves also cause significant challenges to reproducibility. Similarly to some instances of feature creation, certain machine learning models require randomness for training. Common examples of this scenario include tree ensembles, cross validation, and neural networks. Tree ensembles require random feature and data extraction, cross validation relies on random data partitions, and neural networks use randomness to initialize their weights. The randomness causes slight differences between models, even ones with the same training data; these models then won’t meet the requirements of reproducibility (Soledad Galli).

- Another potential problem may arise when working with arrays. Certain APIs used to build models utilize arrays rather than data frames. Unlike data frames, arrays don’t have named features, so ordering of the columns is the only way to reliably identify them. In these cases, programmers and data scientists will need to devote additional attention to ensure to always pass the features in the correct order (Soledad Galli).

- Similar to feature creation, simple solutions can combat most of these threats to reproducibility. Data scientists must give extra care to record orders in which they pass features, the hyper-parameters used, set the seeds when needed, and mind the structure if the final model is an ensemble of models (Soledad Galli).

- 



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
 