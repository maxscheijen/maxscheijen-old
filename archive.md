---
layout: page
title: Post archive
permalink: /archive/
---

This is an overview of the blog posts and their categories.

{% for tag in site.categories %}
  <h3>{{ tag[0] }}</h3>
  <ul>
    {% for post in tag[1] %}
      <li><a href="{{ post.url }}">{{ post.title}}</a></li>
    {% endfor %}
  </ul>
{% endfor %}