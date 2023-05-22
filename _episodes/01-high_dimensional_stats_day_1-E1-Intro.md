---
title: "Exploring High Dimensional Data"
teaching: 20
exercises: 2
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

## Contents of this lesson
1. Describe how high dimensional data visualization and analysis can reveal a research story in noisy data.
9. Exercise - Using your plots or new plots How can you grasp what signal is present in a 13 dimensional dataset? What does each dimension contribute?
10. Discussion: what story or stories seem to be in this data?
4. Explain how to form lower dimensional descriptions/abstractions of high dimensional data
11. Exercise - how would you simplify a dataset?
22. Are there dimensions that don’t seem to add anything?
23. What new representations could you choose?
24.  movie, radio pulses, customer dataset
5. Solution
1. Movie
1. Average color value of every frame’s px
2. reviews of the movie
2. radio signals - just take max amplitude?
3. Customer table - choose specific features?
6. Note these approaches are lossy compression

5. Identify and explain at least one possible method and use-case for reducing dimensionality
    12. PCA concept slides
    13. Visualization webpage: [https://setosa.io/ev/principal-component-analysis/](https://setosa.io/ev/principal-component-analysis/)
    14. Live coding:
        Apply PCA to breast cancer data set
    15. Exercise
    25. Apply PCA to breast cancer data set
    26. Vary parameters
    27. What effects do you notice
6. Describe how PCA can help you tell a story about a high dimensional dataset
16. Show story/signal with PCA result
28. malignant/benign definition.



# Introduction - what is high dimensional data?

### What is data?

### da·ta
/ˈdadə,ˈdādə/
_noun_
"the quantities, characters, or symbols on which operations are performed by a computer"

—Oxford Languages


(how is data formatted? structured, semi-structured, unstructured: flat file, json, raw text)

There is a conversion to numerical representation happening here

A rectangular dataset:
Original data set not rectangular, might require conversion that produces high dimensional rectangular data set.

We’re discussing structured, rectangular data only today.

# What is a dimension?

### di·men·sion
/dəˈmen(t)SH(ə)n,dīˈmen(t)SH(ə)n/

_noun_
noun: __dimension__; plural noun: __dimensions__
1. a measurable extent of some kind, such as length, breadth, depth, or height.
2. an aspect or feature of a situation, problem, or thing.

—Oxford Languages

# A Tabular/Rectangular Data Context
<!-- ![Table](../fig/day_1/tabular_data.png) -->

<center>
<img
    src = "../fig/day_1/tabular_data.png"
    alt = 'A Schematic of the arrangement of Tabular Data with columns/features rows/observations'
/>
</center>


Each *row* is an **observation** is a **sample**

<center>
<img src="../fig/day_1/tabular_data_row_highlight.png"
     />
</center>


Each *column* is a **feature** is a **dimension**

<center>
<img src="../fig/day_1/tabular_data_dim_highlight.png"/>
</center>

The* *index* is not a dimension

<center>
<img src="../fig/day_1/tabular_data_idx_highlight.png"/>
</center>


# Examples of datasets with increasing dimensionality
    
1. There are some number of observations
2. every feature of an observation is a dimension
3. the number of observations is not a dimension
   



### 1 D

1. likert scale question (index: respondent_id, question value(-3 to 3)

### 2 D

1. scatter plot (x, y)
2. two question survey (index: respondent_id, q1 answer, q2 answer)
3. data from temperature logger: (index: logged_value_id, time, value)


### 3 D

1. surface (x, y, z)
2. scatter plot with variable as size per point (x, y, size)
3. [consecutive pulses of CP 1919](https://www.wemadethis.co.uk/blog/2015/03/unknown-pleasures-and-cp-1919/) (time, x, y)
<center>
            <img src="../fig/day_1/CP1919_pulses_crop.png"/>
</center>
4. 2d black and white image (x, y, pixel_value)
5. moves log from a game of 'battleship' (move number, x coord, y coord, hit or not)

## Battle ship moves: 
### discussion point is this 3d or 4d?

is the move number a dimension or an index?


|move_id|column (A-J)|row (1-10)| hit |
| :-: | :-: | :-: | :- |
|0|A|1|False|
|1|J|10|True|
|2|C|7|False|
|n|...|...|





### it's an index!
1. order sequence matters but not the specific value of the move number


### it's a dimension!
1. odd or even tells you which player is making which move
2. order sequence is important, but when a specific moves get made might matter - what if you wanted to analyze moves as a function of game length?


### There is always an index

1. it is an index 
2. that doesn't mean there is no information there
2. you can perform some feature extraction on the index
4. this would up the dimensionality of the inital 2d dataset:
    1. player
    2. player's move number

# consider a short, black and white, silent film, in 4K:

It has the following properties:
1. 1 minute long
1. 25 frames per second
2. [4K resolution](https://en.wikipedia.org/wiki/4K_resolution) is 4096 × 2160.
3. standard color depth 24 bits/pixel



## How many observations are there?

 60 seconds x 25 frames per second = 1500 frames or 'observations'.

## what dimensions could there be per observation?

1. pixel row (0-2159)
2. pixel col (0-4095)
3. pixel grey value (0-255)

## would the dimensions change if the film was longer or shorter?

1. The dimensions would NOT change. 
2. There would simply be a greater or fewer number of 'observations'

## what if it was a color film?

yes - there are more dimensions/features per observation now.

1. pixel row (0-2159)
2. pixel col (0-4095)
3. pixel red value (0-255)
4. pixel green value (0-255)
5. pixel blue value (0-255)

# 4 D
1. surface plus coloration, (x, y, z, color_label)
3. surface change over time (x, y, z, time)

# 30 D


2. Brain connectivity analysis of 30 regions

# 20, 000 D - human gene expression

<center>
<img src="../fig/day_1/DNA_microarray_DrJasonKangNCIFlicker.jpg"/>
</center>

> ## Exercise 1: examine Titanic dataset
> What about the data are you using? or a dataset you know about? e.g. kaggle [Titantic Dataset](https://www.kaggle.com/competitions/titanic/data)?
> 1. What columns are the index?
> 2. What columns are the dimensions?
> 3. how many dimensions are ther?
> 4. is there extra information in the index that could be an additional feature?
> > ## Solution
> > asdf
> {:.solution}
{:.challenge}





##  Exercise 2: imagine building a model to predict survival on the titantic
1. would you use every dimension?
2. what makes a dimension useful?
3. could you remove some dimensions?
4. could you combine some dimensions?
5. how would you combine those dimensions?
6. do you have fewer dimensions after combining?
7. do you have less information after combining?


1. no, some variables are poor predictors and can be ignored
2. if it is (anti-)correlated with survival i.e. has information.

# End of part 1

## We reviewed:

1. what is data
2. what is a dimension
3. an index is not a dimension
4. examples of datasets of different dimensinoality
5. examined dimensionality of the titantic dataset

in part two we'll start exploring a new dataset