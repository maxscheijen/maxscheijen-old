---
layout: post
title: "Estimating COVID-19's $R_t$ in Real-Time for the Nethelands"
categories: [data-science, baysian statistics, covid-19]
author: Max Scheijen
---

The effective reproduction number $R_t$ is a way to measure how quickly any virus spreads. In this article, I implement a simple Bayesian model to estimate the $R_t$ in real-time in the Netherlands. This simple Bayesian model developed by Systrom et al. {% cite rtlive2020 %}. In contrast to Systrom et al, my model uses new daily COVID-19 cases to estimate $R_t$ in real-time.

<div class="alert alert-warning">
  <strong>Note: </strong> All credit for developing this model goes to Systrom et al. {% cite rtlive2020 %}. They did all the hard work. I simply applied his insights, research, and model to estimate $R_t$ in real-time in the Netherlands.
</div>

The plot below displays an estimate of $R_t$ estimated in real-time in the Netherlands. The reproduction number estimates is a measure of the number of people infected by one contagious person.

![original_smoothed](https://raw.githubusercontent.com/maxscheijen/rt-live-netherlands/master/figures/most_likely_rt.svg)

The estimation of $R_t$ is based on the smoothed new case count. As suggest by Systrom et al., I apply a gaussian filter to the daily new case count time series because of the stochastic nature, caused by late reporting or corrections. The choice the size of the filter arbitrary, but the real-world process is not nearly as stochastic as the actual reporting, smoothing helps with this.

Systrom's more complex Bayesian model uses the positive test rate to estimate $R_t$. Ideally, I would like to use this model. However, [the Rijksinstituut voor Volksgezondheid en Milieu](https://www.rivm.nl/) (RIVM) and the [Gemeentelijke gezondheidsdiensten](https://www.ggd.nl/) (GDD's) only publishes the positive test rate weekly. The Bayesian model in this article is simply the implementation for the first iteration of the Systrom et al. approach. If these organizations decide at a later time to publish the percentage of positive tests per day, I will consider implementing the more complex Bayesian model. Also, I will try to implement the more complex Bayesian model that uses MCMC sampling at a later time based on new daily case counts.

The data used to create this model is gathered by De Bruin {% cite de_bruin_2020 %} based on the raw data provided by the RIVM.

## References

{% bibliography --cited %}
