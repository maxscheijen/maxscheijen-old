---
layout: post
title: "Text Mining Dutch News Articles"
categories: [nlp, data analysis, deployment]
author: Max Scheijen
---

In this article, I analyze Dutch news articles. I gathered the news articles by scraping them from the NOS <a href="https://nos.nl/" target="_blank">website</a>. The NOS is one of the organizations of the public broadcasting system in the Netherlands and established in the Media Act. Its task is to provide the media supply for the national public media service in the field of news, sports, and events. The NOS is also one of the biggest online news websites in the Netherlands.

As of the publishing of this post, the dataset contains 218,609 news articles spanning more than 10 years (the first article in the archive is published on 2020-01-01). In addition to the article content, the dataset contains the article title, the category, and the publishing grade. The NOS also publishes liveblogs. I do not consider these news articles. Therefore they are excluded from the dataset.

Before analyzing the news articles, the raw data needs to be cleaned. The texts contain a lot of white spaces and some weird characters. I use regular expressions and some Python string methods to remove extra white spaces and these wrong characters. After cleaning the content of an article will look something like this:

> 'Spanje is met ingang van vandaag voorzitter van de EU. De Zweedse premier Fredrik Reinfeldt heeft het stokje, formeel om middernacht, overgedragen aan zijn Spaanse collega José Luis Rodriguez Zapatero.  Spanje is het eerste land dat het roulerend voorzitterschap overneemt onder het Verdrag van Lissabon, dat op 1 december in werking is getreden. Nieuwe functies  De rol van het voorzitterschap is met het in werking treden van het Verdrag van Lissabon veranderd. Voortaan zal de Belg Herman van Rompuy de vergaderingen van de Europese Raad voorzitten. Van Rompuy vertegenwoordigt de EU ook internationaal, samen met de Britse Catherine Ashton. Zij is de buitenlandminister van de EU, ook een nieuwe functie. Spanje heeft het economisch herstel hoog op de agenda van de Europese Unie gezet. Van Rompuy organiseert volgende maand een extra EU-top over de aanpak van de economische crisis. Geslaagd De Zweden mogen terugzien op een geslaagd voorzitterschap. In het afgelopen half jaar kwam het nieuwe Verdrag er, werden er topbenoemingen geregeld en de kredietcrisis bestreden. Alleen de klimaattop in Kopenhagen was een project dat minder succesvol werd afgesloten. '

Most of the analysis focuses on the differences between article features based on the category they are classified as.

## Article Counts

The first thing to look at is the article count over time between 2010 and 2020. To get a less stochastic article count, I decided to group them by months over time. From 2010 to the middle of 2016, the monthly article count increased from around 1000 to a little more than 2000. After 2016 we see a downwards trend in the number of monthly articles published to 1100 in 2020.

![png](https://maxscheijen.github.io/assets/img/nos_analysis_blog_files/nos_analysis_blog_9_0.png)

More interesting is probably to look at the change of monthly article count when we split the articles based on category. If we look at the absolute count over time, we see the same trend as the overall article count. This is most likely caused by the "Binnenland" (domestic) and the "Buitenland" (foreign) categories, which display the same tends.

![png](https://maxscheijen.github.io/assets/img/nos_analysis_blog_files/nos_analysis_blog_11_0.png)

Therefore I choose to normalize the counts based on category. The second graph shows the share of a news category in the total number of articles in a specific month. For example, there is a large bump in the domestic news share (and drop in foreign news) in the first quarter of 2020 caused by the initial Coronavirus outbreak. In contrast, most of 2011, the foreign news articles were most prevalent, probably caused by the start of the <a href="https://en.wikipedia.org/wiki/Arab_Spring" target="_blank">Arabic Spring</a>, the <a href="https://en.wikipedia.org/wiki/Killing_of_Osama_bin_Laden" target="_blank"> capture of Osama Binladen</a>, and the <a href="https://en.wikipedia.org/wiki/2011_Norway_attacks" target="_blank">terrorist attacks in Norway</a>.

## Word Count

The word count of an article can be an indicator of the quality of the news article. Blumenstock et al. {% cite blumenstock %} show that longer Wikipedia articles are more likely of better quality. Longer articles are more often featured than shorter Wikipedia articles. They state that word count is a robust method of article quality.

Let us first look at the overall word count of all the articles published by the NOS. As expected, the word count of the article is heavily right-skewed (when used for modeling probably better to use the log). Also, the word count of the title is more normally distributed, which is expected.

![png](https://maxscheijen.github.io/assets/img/nos_analysis_blog_files/nos_analysis_blog_15_0.png)

Tech and Culture & Media have the highest median word count. This makes sense to me. These articles are often less time-dependent. Meaning that their publishing is less dependent on the daily news cycle. They are still relevant at a later moment than political news. This suggests that writers have more time to write these articles, can incorporate more detail, leading to longer pieces. The "remarkable" and "regional news" category have the lowest median news count. This is expected most of these articles report small news events, leading to shorter pieces.

![png](https://maxscheijen.github.io/assets/img/nos_analysis_blog_files/nos_analysis_blog_18_0.png)

## Readability

Another important of a news article is how easy it is to read. Readability measures how hard it is to read a text. There are several methods to calculate readability. These methods are not that reliable, but they give some insight into the readability of a written text. In this analysis, I use the Flesch reading-ease test to calculate readability. Douma {% cite douma_1960 %} slightly tweaked the values in the Flesch reading-ease test formula such that the measures can also be used for Dutch texts.

<details>
  <summary><b>More on the calculation and interpretation of the Flesch-Douma formula</b></summary>
  The Flesch-Douma formula {% cite douma_1960 %} equates readability to the following formula:
  $$ \text{readability} = 206.835 - 0.93 \bigg{(}\frac{\text{total words}}{\text{total sentences}}\bigg{)} - 77.0 \bigg{(}\frac{\text{total syllables}}{\text{total words}}\bigg{)}$$

  The readability score is related to education level in the following manner:

  <table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">Score</th>
      <th class="tg-0pky">School level</th>
      <th class="tg-0pky">Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="tg-0pky">100–90</td>
      <td class="tg-0pky">5th grade</td>
      <td class="tg-0pky">Very easy to read. Easily understood by an average 11-year-old student.</td>
    </tr>
    <tr>
      <td class="tg-0pky">90–80</td>
      <td class="tg-0pky">6th grade</td>
      <td class="tg-0pky">Easy to read. Conversational English for consumers.</td>
    </tr>
    <tr>
      <td class="tg-0pky">80-70</td>
      <td class="tg-0pky">7th grade</td>
      <td class="tg-0pky">Fairly easy to read.</td>
    </tr>
    <tr>
      <td class="tg-0pky">70-60</td>
      <td class="tg-0pky">8th &amp; 9th grade</td>
      <td class="tg-0pky">Plain English. Easily understood by 13- to 15-year-old students.</td>
    </tr>
    <tr>
      <td class="tg-0pky">60-50</td>
      <td class="tg-0pky">10th to 12th grade</td>
      <td class="tg-0pky">Fairly difficult to read.</td>
    </tr>
    <tr>
      <td class="tg-0pky">50-30</td>
      <td class="tg-0pky">College</td>
      <td class="tg-0pky">Difficult to read.</td>
    </tr>
    <tr>
      <td class="tg-0pky">30-10</td>
      <td class="tg-0pky">College graduate</td>
      <td class="tg-0pky">Very difficult to read. Best understood by university graduates.</td>
    </tr>
    <tr>
      <td class="tg-0pky">10-0</td>
      <td class="tg-0pky">Professional</td>
      <td class="tg-0pky">Extremely difficult to read. Best understood by university graduates.</td>
    </tr>
  </tbody>
  </table>
</details>

The articles categorized as Foreign, Politics, Tech, and Economy are the heardest to read with a readability score below 60, which is about 10th to 12th-grade-level reading.  So somewhat hard to read.  Categories "remarkable news", "regional news" and "culture & media" are easier to read with a score above 60. However, the overall median scores are not that different across categories.

![png](https://maxscheijen.github.io/assets/img/nos_analysis_blog_files/nos_analysis_blog_22_0.png)

## Sentiment, Polarity, and Subjectivity

Let us look at the sentiment of the texts. Are they positive or negative? Furthermore, we can also analyze the texts for subjectivity. When looking at all the articles together, we see that most of the news articles are neutral in their polarity. Most of the articles hover somewhere in the middle between subjective and objective in their subjectivity measure.

![png](https://maxscheijen.github.io/assets/img/nos_analysis_blog_files/nos_analysis_blog_26_0.png)

Even if we parse the articles by news category we do not see many differences in the distributions of polarity and subjectivity scores.

![png](https://maxscheijen.github.io/assets/img/nos_analysis_blog_files/nos_analysis_blog_28_0.png)

## Informative words

We can use TF-IDF, short for term frequency-inverse document frequency, to extract the most informative words out of every article. TF-IDF  is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. In this case, I removed stop words, because most of the time they don't contain any useful information. Also, I ignore terms that appear in less than 0.5% of the documents. Let's see if what the most informative terms are in the news articles in every category.

![png](https://maxscheijen.github.io/assets/img/nos_analysis_blog_files/nos_analysis_blog_35_0.png)

The TF-IDF result is pretty interesting. For example, we can see that the most informative terms (translated to English) in articles categorized as politics are: chamber, cabinet, minister, political parties (VVD, PvdA, CDA, D66), member of parliament, and secretary of state. All words that are associated with politics. The most informative terms in the economic news articles are also associated with the economy. For example, percent, Euro, company, billion, banks, and money. In the Royal Family category, we see that the words King, Queen, Prince, Princess, Willem (the king's first name) are all highly informative.

The code that generated the plot and scarped the data can be found on <a href="https://github.com/maxscheijen/blog-post-code/tree/master/nos-analysis" target="_blank">Github</a>.

## References

{% bibliography --cited %}
