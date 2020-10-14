---
layout: post
title: "Estimating COVID-19's $R_t$ in Real-Time for the Netherlands"
categories: [data-science, bayesian statistics, covid-19]
author: Max Scheijen
---

The <a href="https://en.wikipedia.org/wiki/Basic_reproduction_number#Effective_reproduction_number" target="_blank">effective reproduction number</a> $R_t$ is a way to quantify how quickly a virus spreads and measures the number of people infected by one contagious person. In this post, I model the Dutch daily case count with a simple Bayesian model proposed by Systrom et al. {% cite rtlive2020 %} that allows us to estimate the $ R_t $ in real-time.

The **daily updated** graph below displays the real-time estimate of the $R_t$ in the Netherlands which is obtained by modelling the daily national case count data gathered by De Bruin {% cite de_bruin_2020 %}.

![rt_live](https://raw.githubusercontent.com/maxscheijen/rt-live-netherlands/master/figures/most_likely_rt.svg)

The estimation of $R_t$ is based on the smoothed daily case count. As suggest by Systrom et al. {% cite rtlive2020 %}, a gaussian filter is applied to the daily case count time-series. This reduces the noise of the stochastic nature of case counts, caused by late reporting or corrections. A window size of 7 days is arbitrarily chosen to smooth this stochastic process. This gives late-stage reporting, back-logs and corrections a week to catch up. The real-world infections are much less stochastic than the process of reporting, making smoothing necessary. The smoothing applied to the daily case count displayed in the figure below.

![original_smoothed](https://raw.githubusercontent.com/maxscheijen/rt-live-netherlands/master/figures/original_smoothed.svg)

<div class="alert alert-warning">
  <strong>Note: </strong> All credit for developing this model goes to Systrom et al. {% cite rtlive2020 %}. They did all the hard work. I simply applied their insights, research, and model to estimate the real-time $R_t$ in the Netherlands.
</div>

Later on, Systrom et al.'s {% cite rtlive2020 %} developed a more complex Bayesian model that uses the positive test rate and <a href="https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo" target="_blank">MCMC</a> sampling to estimate $R_t$, as seen on <a href="https://rt.live" target="_blank">rt.live</a>. The positive test rate is a more useful measure because the number of raw positive tests is influenced by how many people are tested. The more you test, the more positive cases you will find. Therefore making test positive rate more informative.

Furthermore, this simple model does not take into account the onset distribution or the incubation period of the virus. More plainly stated this is the distribution of the delay between infection and confirmed positive test. The more complex Bayesian model does take this into account.

Ideally, I would like to use this more complex model. However, the <a href="https://www.rivm.nl/en/novel-coronavirus-covid-19/current-informatio" target="_blank">Rijksinstituut voor Volksgezondheid en Milieu</a> (RIVM) and the <a href="https://www.ggd.nl/" target="_blank">Gemeentelijke gezondheidsdiensten</a> (GGD's) only publish the positive test rate weekly. If these organizations decided to publish the daily positive test rate, I consider implementing the more complex model.

Code can be found on <a href="https://github.com/maxscheijen/rt-live-netherlands" target="_blank">GitHub</a>.

## References

{% bibliography --cited %}
