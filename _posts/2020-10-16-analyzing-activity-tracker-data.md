---
layout: post
title: "Analyzing activity tracker data"
categories: [data-science, analysis, statistics]
author: Max Scheijen
---

I have been wearing a Fitbit Charge 3 activity tracker for a little more than a year now. This fitness tracker does more than simply track steps. Because of its additional sensors, it can track more movements and physical attributes. For example, it also tracks distance, keeps track of your heart rate, the calories you burn during the day, and tracks your sleep.

In this post, I analyze the data gathered by my tracker over the course of one year. This gives me some insight into my physical activities this year. Most activity or fitness tracker providers allow you to download your data. Fitbit gives you your data in different formats such as JSON and CSV. Sometimes all data is put into one file, sometimes in multiple files. I chose to stitch these files together, for easier analysis. Not every measurement is recorded at the same rate. Therefore I often resample the data to make it easier to analyze and compare.

## Steps

Let's first take a look at the most basic levels of activity tracking, the step count. The Fitbit Charge 3 logs the number of steps taken every minute using a 3-axis accelerometer. This accelerometer allows the device to determine the frequency, duration, intensity, and patterns of the movements you make. If GPS is not enabled, distance is estimated using steps and stride length.

Let's resample the step data from minute level measurements to hourly level measurements. This takes me from 261184 samples to 8688 samples, making the data less noisy. The distribution of hourly steps can be seen in the plot below. As expected, most 1 hourly steps fall into the bin containing 0 steps per hour. This can be explained, by the hours I sleep and, I do not take any strides. Furthermore, most of my time during the day I spend behind a desk, and I don't make many movements.

![png]({{ site.url }}/assets/img/2020-10-16-analyzing-activity-tracker-data_5_0.png)

This makes me question if my step behavior changed over time? Let's plot a time series of my step data. Resampling to hourly data is not enough because there just t too many samples. Therefore I chose to resample to daily data. This makes it also easier to see trends. Gaussian smoothing also makes it easier to see trends, making the data less noisy. I use a window size of seven, which basically smooths data to weeks. The plot below displays the time-series step count over the last year. There is an upwards trend from early March to the middle of June, which can be explained by the fact that I started running. Also, notice the bump around the start of September. This bump can be explained by my holiday in which I walked a lot.

![png]({{ site.url }}/assets/img/2020-10-16-analyzing-activity-tracker-data_8_0.png)

After looking at the less fine-grained data, let's take a look at the minute level data. We take the average step count data every minute over all the days in the year and visualize it. Let's also plot the overall mean step count every minute over the year. See the graph below for a visualization of this data. Notice something strange? There is a large bump around minute 50 every hour. There is a clear explanation for this bump. The Fitbit Charge 3 gives a vibrating notification every 10 minutes before the hour if you haven't taken 250 steps that specific our. This notification clearly gets me to move, making me change my behavior.

![png]({{ site.url }}/assets/img/2020-10-16-analyzing-activity-tracker-data_11_0.png)

## Heart rate

As earlier stated, the activity tracker also measures heart rate. The heart rate is logged every 5 seconds. Over a year, this gives us 4,011,312 samples. Heart rate is measured in BPM, or beats per minute. The tracker also records a "confidence" variable. This measures how confident the tracker is in its heart rate measurement.

Because of the fine-grained data, I again resample the data to minutes. Let's make somewhat the same plot as we did with steps. However, instead of minute data, we use hourly data. This gives us a matrix of size 365x24. The data is plotted in the graph below.  Notice the lower heart rate during the night, which is understandable when sleeping.

![png]({{ site.url }}/assets/img/2020-10-16-analyzing-activity-tracker-data_14_0.png)

Did my average heart rate change over time? The plot below displays that my mean daily hearth rate did not really change over time. Note this is not my resting heart rate.

![png]({{ site.url }}/assets/img/2020-10-16-analyzing-activity-tracker-data_17_0.png)

## Calories burned

The tracker estimates the calories burned using the basal metabolic rate (BMR). This is the rate at which you burn calories at rest to maintain vital body functions like breading, blood circulation, and heartbeats. Heartbeat is also included in this measurement to measure calories burned during exercises. The BMR is based on physical measurements such as height, weight, sex, and age. These physical measurements are used to calculate the number of calories you burn in rest,  such as sleep or when you are not moving. In this rest state, you burn about half of the total calories burned each day.

Let's look at the relationship between calories burned and heart rate. The graph below displays a scatter plot with the heart rate and calories every 30 minutes. There is some small linear relationship between the number of steps and burned calories. However, the strength of the relationship is less than I suspected.

![png]({{ site.url }}/assets/img/2020-10-16-analyzing-activity-tracker-data_21_0.png)

## Sleep

The tracker also measures sleep. The sleep score helps you understand your sleep each night. This allows us to see trends in sleep patterns. Getting a night of good high-quality sleep can have a positive impact on energy, activity, mood, and weight. The tracker calculates your sleep score is based on your heart rate, the time you spend awake or restless, and your sleep stages. My first instinct is that when someone spends more time in deep sleep, the higher the overall sleep score. Let's explore this. Below we see the relationship between the number of deep sleep minutes and Sleep score. Furthermore, the plot also displays the distribution of deep sleep minutes and sleep scores.

![png]({{ site.url }}/assets/img/2020-10-16-analyzing-activity-tracker-data_25_0.png)

Because the time I spend in deep sleep doesn't seem to be highly correlated with sleep score, I try to answer the a different question: which sleep-related measurement is mostly correlated with a high sleep score? The plot below displays the correlation between all the different features and sleep scores.

![png]({{ site.url }}/assets/img/2020-10-16-analyzing-activity-tracker-data_29_0.svg)

Most of these measures make sense in influence the sleep score negatively or positively. Duration or minutes of sleep has a positive impact, as do most of the sleep-related measurements. The measurements related to being awake correlate hurt the sleep score.
