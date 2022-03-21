---
title: "Exploring High Dimensional Data"
teaching: 0
exercises: 5
questions:
- "Key question (FIXME)"
objectives:
- "Provide intellectual access to discussions of information-age high dimensional data(sets)"
- "Define, identify, and give examples of high dimensional datasets"
- "Summarize the dimensionality of a dataset"
- "Explain best practices for how to organize / structure high dim data for reuse"
- "Demonstrate at least one method to visualize, and explore a high-dimensional dataset"
- "Describe how high dimensional data visualization and analysis can reveal a research story in noisy data."
- "Explain how to form lower dimensional descriptions/abstractions of high dimensional data"
- "Identify and explain at least one possible method and use-case for reducing dimensionality"

keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
---
FIXME

{% include links.md %}

# Introduction - what is high dimensional data?
- what is data? 'the quantities, characters, or symbols on which operations are performed by a computer' - literally anything.
- (how is it formatted? structured, semi-structured, unstructured: flat file, json, raw text)
- what is a dimension? - 'A measurable extent of some kind', 'an aspect of feature of a situation, problem, or thing' (Oxford Languages)
- 1d data set - number line, likert scale
- 2d data set - scatter plot, 2 columns, survey answer
- 3d data set - surface, scatter plot, consecutive pulses of CP 1919, 2d black and white image data set
- 4d data set - surface plus coloration, 2d full color image data set, a movie
- 30 dimensional dataset - customer feature table
- 1000 dimensional dataset - gene expression 

> ## Exercise - how would you simplify a dataset? What new representations could you choose?
> movie, radio pulses, customer dataset
> > ## Solution
> >  movie - average every still's x and or y coords?, use  reviews of the movie instead
> >  radio signals - just take max amplitude?
> >  Customer table - choose specific features?
> {: .solution}
{: .challenge}

Discuss what dim reduction/ simplication is doing - lossy compression, may reveal signal

> ## Exercise - how many dimensions are in this data set? what are the labels?
> open cancer data set, explore
> > ## Solution
> >  data set has 13 dimensions, all quantifcations of tumors benign and malignant.
> >  
> {: .solution}
{: .challenge}

> ## Exercise
> How can you grasp what signal is present in a 13 dimensional dataset? What does each dimension contribute? 

> > ## Solution
> >  - correlation matrix heat map
> >  - scatter matrix
> >  - facet plots
> {: .solution}
{: .challenge}


> ## Exercise
> Some things can be learned. What is there are many more dimensions? 5000 dimensions?
> > ## Solution
> >  - load Dorothea drug discovery data set
> >  - have to subset the data to get comprehensible results
> {: .solution}
{: .challenge}

# challenges discussion
1. what was still easy?
2. what was difficult?
3. did you find a story or signal?
4. Where is your own research dataset on this field?


# Principle Component Analysis Intro
## Frame of Reference
Mathematical tool for  finding the most meaningful _frame of reference_ or _basis_
show simple 2d example with rotated FOR. 
first two axes are describing the data. now one axis describes all the variability in the data
*2d data set now a 1d dataset!*
a different FOR simplified the data.
Naive FOR or Naive Basis -> change of basis -> Transformed FOR/Basis

## Simple explanation - how did that happen?
1. PCA finds the dimension of highest variance, that's p1
2. Then define the dim perpendicular to p1 with highest variance as p2
3. up to pn.
4. construct new basis from those assignments.
show chart - original 2d blob of points from above with p1, p2 as X', Y'

## Caveats - note that PCA assumes:
1. PCA assumes variance noise << variance signal, i.e. low noise.
*if all your data is noisy this isn't going to work very well*
	- demonstrate this with two side by side examples.
2. output data is now in terms of _prinicipal components_ not your original dimensions - this may or may not matter to you.

## Additional reading: 
1. A Tutorial on Prinicipal Component Analysis, J. Shlens (2014) https://arxiv.org/pdf/1404.1100.pdf 
2. I.T. Jolliffe (2002) Principal Component Analysis, Second Edition

## additional visualization
https://setosa.io/ev/principal-component-analysis/

> ## Exercise
> PCA explore with synthetic data
> > ## Solution
> >  - how does PCA handle data with a large number (n=100) dimensions?
> >  - how does PCA handle some dimensions with high noise?
> >  - how does PCA handle all dimensions with high noise?
> {: .solution}
{: .challenge}


> ## Exercise
> PCA explore with WI breast cancer data
> > ## Solution
> >  - compare output to starting results
> >  - what is gained?
> >  - what is lost?
> {: .solution}
{: .challenge}