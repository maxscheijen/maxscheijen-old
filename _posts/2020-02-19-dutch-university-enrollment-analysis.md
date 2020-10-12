---
layout: post
title: "Dutch University Enrollment Analysis"
categories: [data-science, social-science, analysis, visualization]
author: Max Scheijen
---


In this post, I'll do a quick descriptive analysis of some enrollment data from Dutch universities. I won't do any predictive modeling or statistical testing. The data is provided by the <a href="https://duo.nl/open_onderwijsdata/databestanden/ho/ingeschreven/wo-ingeschr/ingeschrevenen-wo1.jsp" target="_blank">Dienst Uitvoering Onderwijs</a>, which is the executive organization for education of the Dutch government. The data ranges from 2014-2018. 

This post is written in English. However, some of the labels in the figures are written in Dutch because the dataset contains categories written in this language. However, they aren't too hard to figure out!

This post is structured in the following way: First, we investigate the distribution of academic degrees. Secondly, we look at the differences in faculty enrollment between faculties and over time. After that, we analyze the geographical distribution of enrolled students. Last, I examine the differences between gender ratios regarding individual universities and study programs.

<div class="alert" style="background-color: #f6f6f6">
    <strong>Note: </strong> Some "hogescholen" offer master degrees these schools are however <u>dropped</u>.
</div>

## Academic Degrees
My first instinct is to analyze the count distribution between academic degrees. Netherlands has two general academic degrees. A bachelor degree and a master degree. The most bachelor degrees in take three years to obtain. An academic master's degree is a one-year or two-year course (and in exceptional cases more than two years long).

![png]({{ site.url }}/assets/img/2020-02-19-academic-degree.png)

As the figure above displays, most students enroll in a master's degree between 2014 and 2018.  However, if we take the bachelor's degrees together with the "propedeuse bachelor" degree, which is obtained after completion of the first-year bachelor, this category becomes most frequent.

## Faculty Enrollment
A faculty is a division within a university comprising one subject area or a group of related subject areas. Let's look at the differences of enrollment between faculties.

![png]({{ site.url }}/assets/img/2020-02-19-enrollment-faculty.png)

The faculty "Gedrag en Maatschappij" (across all universities) has the most students enrolled between 2014 and 2018. Most Social and Behavioral Sciences are part of this faculty. As stated by <a href="https://www.uva.nl/en/about-the-uva/organisation/faculties/faculty-of-social-and-behavioural-sciences/organisation-and-contact/organisation.html" target="_blank">The Faculty of Social and Behavioural Sciences of the University of Amsterdam</a>: 

>"_Research and education at the Faculty address societal and human behavior related themes, like the impact of new media on society, healthcare, urbanization, human and child development, mental health, inequalities, diversity and social cohesion_."

Some studies in this faculty are social sciences such as 
Anthropology, Economics (sometimes), Political Science, Psychology and Sociology and many more.

There is a large shortage of teachers in the Netherlands. This figure shows that in the 2014-2018 period, enrollments at the education faculties were, in absolute terms, the lowest compared to other faculties. However, a connection between the two observations cannot be established.

We can, however, try to find out whether there have been changes in the number of enrollments over time. We do this by comparing the 2014 registrations with those of 2018. 

![png]({{ site.url }}/assets/img/2020-02-19-enrollment-faculty-overtime.png)

We can see that almost all faculty enrollments increased between 2014 and 2018. The biggest absolute increases are for the "Techniek" like Engineering and Computer Science programs and "Natuur" like Physics, Biology and Chemistry programs. Both consist of so-called "exact" or "hard" sciences. There is a lot of focus to increase technical studies in the Netherlands. Which seems to be working! 

The only faculty that has a decrease in enrollments is the education faculty. This can be one of the explanations for the teacher shortage.

## Enrollment based on location and institution
Let's now look at the number of enrollments based on the location. We have both city and province/state data. Not every province has a university. Therefore not all provinces will be represented in the graph below. 

![png]({{ site.url }}/assets/img/2020-02-19-enrollment-province.png)

We see that both Zuid-Holland and Noord-Holland Universities have the most students enrolled. Both provinces also have the biggest populations what can be an explanation. Zuid-Holland as 3 universities and Noord-Holland has 2 universities. 

Location can also be tied to the enrollment numbers for individual Universities. 

![png]({{ site.url }}/assets/img/2020-02-19-enrollment-university.png)

We can see that the biggest enrollment universities are all located in the so-called "Randstad". This is a metropolitan area that includes the four largest cities in the Netherlands: Amsterdam, Rotterdam, The Hague, and Utrecht.

## Gender and Enrollment
Another variable included in the dataset is gender. We'll look at ratios between the enrollment of men and women at universities and fields of study.

The figure below displays the absolute ration of enrollments based on gender at universities in the Netherlands between 2014 and 2018. 

![png]({{ site.url }}/assets/img/2020-02-19-enrollment-gender-university.png)

We can see that at almost all universities women are in the majority. The only universities where men have a clear majority are technical universities.

Let's look at the fields of study that have the biggest ratio of differences between men and women and vice versa. I excluded programs that have less than 100 enrollments over 4 years to keep some robustness. 

![png]({{ site.url }}/assets/img/2020-02-19-enrollment-gender-study.png)

<sup>* DPiEaCS (research): Developmental Psychopathology in Education and Child Studies (research)</sup><br>
<sup>* DaSiCaA (research): Development and Socialisation in Childhood and Adolescence (research)</sup>

Most of the technical programs have a high ratio of men relative to women.  Automatic Technology has a ration of 42 to 1. Meaning that for every 42 men following the program 1 woman enrolled. On the other hand the non-research program Education and Child Studies as about a women ratio of 22 to 1. 

These programs seem divided along with the "traditional"  view of gender divided fields of study. Men have the enrollment overhand in technical and computer science programs and women tend to have the overhand in programs focuses on education and health. 

## Conclusion
In this post, I did a quick and basic analysis of University enrollment data. There are a lot more ways to analyze this data.  However, we saw that social studies programs have the highest enrollment in the period between 2014 and 2018.  That technical studies are growing fastest. Most enrolled students study at Universities in the "Randstad" and that there seem to be some traditional gender divisions for some programs.  

<i>This code to generate do this analysis and generate the figures is on available on <a href="https://github.com/maxscheijen/blog-post-code/tree/master/dutch-university-enrollment-analysis" target="_blank">GitHub</a>.</i>
