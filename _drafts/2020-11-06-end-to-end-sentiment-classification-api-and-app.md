---
layout: post
title: "End to end Sentiment Classification API and APP"
categories: [nlp, classification, deployment]
author: Max Scheijen
---

In this article I would like demonstrate a end-to-end machine learning project. How do we create a simple sentiment classifier, and deploy it on Herkuo. We also create a simple Streamlit app which allows the user to interact with the sentiment classifier and get some more information about the why the text is classifier postivuly or negatively. Furthermore more we also build a simple API in flask that can be used by other services to rectrieve information from our classifier.

So what are we going to build. We want to train a classifier that is able to predict the sentiment from a input text. The classifier needs to be able to predict wether a text is positive or negative. Furthermore we want to give the user some information about how the words in there text influence the sentiment prediction. Secondly, we want to be able to deploy the trained model using a Web App make interacting with easy. However, we also want to create an API to which we can post requests and get a prediction response back.

We need to take several steps along the way. I try to apply some basic software engineering practices to make our code modular and easy to understand. This also allows us to quickly change the model architecture or algorithm, use different params without introducting bugs. First we need to collect data, this data needs to contain a text and some label indicating the sentiment of that particular text. After we collect the raw text files, I want to create a train and test dataset this allows us to evaluate our sentiment classification model. When we have created these datasets we need to do some text processing. We take some simple text processing steps nothing fancy. Now we are ready to train our classification model. We implement a logistic regression classifier into a SentimentClassifier object, which makes it easy to change to an other classification algorithm later on. After training the model I implement cross validation to asses the preformance of our model across several metrics. 

Our training is done, now we need to create ways to interact with our trained model. First we create a simple API using flask, which is able to recieve POST requests and give a sentiment prediction as a response. This API allows external or internal services to interact with out sentiment classification model. However, we also want a visual way to interact with our sentiment classifier. Therefore we create a simple Web App using Streamlit, in which allows us to input our own text and get sentiment prediction back plus some addition information on how this prediction came to be.

## Product


Let us first disucss the product side of this project. What is exactly the problem and what are we trying to solve? I will do this by illustrating using hypothetical scenario which we use to reason about the design and implementation process. This will help us to create a modular and configurable model, which can be improved itervarly over time. 

> **Hypothetical scenario**: You work for a company, and they want you to design a system to asses the reputation of the company brand in order to better protect it. The company has a system inplace to scrape social media and review sites based on your company's name. 

Keeping our scenario in mind whay should we think carefully about? Here are some steps to keep in mind when creating our initial machine learning model.

1. First we should identify the problem. 
2. After this we should design a solution with contraints. 
3. We should think about how to evaluate the solution.
4. how we iterate over our solution to make it beter.

### Objective 

The problem statement should not have to do with machine learning. "We do not have a machine learning system to classify the sentiment of mention" is not a correct problem statement. The company has a system that scrapes websites to retrieve reveiws where are organization is mentioned. However currently we do not have the capability to asses if these mentions are positive or negative. You know that a bad reputation online or bad reviews can snowball and the longer you do not adress these the worse the situation will become. 

Based on the problem statement we can forumlate clear objectives. These do not need to be technical at all. These are pure highlevel objectives. In our scenario these could be potentional objective:

> **Objective**: Protect our brand reputation by classifying if the text containing the mention is positie or negative.

However, translating this objective to a working system can be difficult. How do we protect the brand. This will be addressed in the solution.

We should also keep in mind that we need some way to know how these predictions are made, on the basis of what words or combination of words are texts classified as positive or negative?

### Solution

Because we get alot of reviews these can not be labeled manually. We need an automated system to classiy wether these reviews are mainly positive or mainly negative. 

In my opinion the **first model should be kept as simple** as possible. This initial model often provides the biggest boost or improvement to your product. Also make sure you get most of you machine learning infrastructure right using this simple model. This makes iterativy improving your model much easier, and makes the probability of bugs occuring lower.

We already have a systems that scrapes reviews or texts about our company from social medial websites. We have review data. However we don't have a system that can classify wether or not these texts are positive or negative. If we are able to identify negative text, we can adress this customer, help them with their potential problem and keep up our brands reputation. The first solution would be to use the star rating system. Basically the star rating system is an indicator for the sentiment of the task. A 1 star rating is most likely a negative review, and a 5-star rating is most likely a positive review.

Keep our objective in mind these are the things are machine learning needs to be able to preform:

1. Enable employees to determine the sentiment of text concerning our brand.
2. Enable employees to see how confident the system is in its prediction.
3. Enable employees to see why the system predicts a certain sentiment 

I want a **modular pipeline**. Meaning we can change our classification algorithms or add and change text processing transformations easily, without easily breaking the system. It would also be desirable that we can tune some global parameters of the current pipeline using a **config** file. All global configs are stored into a single easy accessable file.

## Setup

So let's setup our project. This allows our machine learning package to be installed as a python package.

```python
from setuptools import setup, find_packages

setup()
```

## Data Collection

Getting good quality data has a large impact on the preformance of your model. If data is hard recieve from within you organization I always encourge you to get some data that is representative of your own data somewhere online.

## Dataset Creation

We have now gather all the raw text files with a corresponding label. We need gather them into one file, which makes it easier to work with down the line.

## Text Preprocessing

It is now time for pre-processing the text. As I earlier stated it is important to implement a simple solution first. Get the pipeline and infrastucture working reliabily firstly using a simple model. After this is done you can focus on more complex systems and models.

The most simple solution to convert text to documents to a numerical representation is to convert the collection of documents into matrix of token (word) counts. We can use scikit-learns `CountVectorizer` to create matrix of token counts.

```python
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()
```

This count vectorizer returns a sparse matrix, which should be taken into account when adding additional processing steps.

## Model Training

We keep our first model simple! Let's use a logistic regression model as our classifier. However let's embed it into custom `SentimentClassifier` object for better modularity.

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
```

## Model Evaulation

## API Creation

## Web App Creation

## Delopyment

