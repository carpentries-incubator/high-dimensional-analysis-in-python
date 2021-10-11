---
title: "High-dimensional Modeling and Feature Selection"
teaching: 0
exercises: 3
questions:
- "Key question (FIXME)"
objectives:
- "Understand the challenges associated with modeling high-dimensional data"
- "Understand the importance of feature selection as a tool for modeling high-dimensional data"
- "Identify and understand some of the possible ways to perform feature selection"
- 
keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
---
FIXME

{% include links.md %}

# Introduction

> ## Discussion
> In this example, we would like to classify images of cats versus dogs. In every image example, a cat or a dog appears at the center of the image with some background imagery present as well. There are two example images provided below. Instead of training our model on every pixel present in each image, what could we do to help the model hone in on the important aspects of the images that relate to how dogs and cats differ?
> 
>
> > ## Solution
> >  - Include only the center of each image--where a dog or a cat appears
> >  - Include only pixels that contain the head of the animal--where differences are more noticeable between the species.
> {: .solution}
{: .challenge}

# Automated Feature Selection
What if don't know which features are important?
