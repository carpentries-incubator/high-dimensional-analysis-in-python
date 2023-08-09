---
layout: page
title: "Setup"
---

## Setup Project Folder
1. Create a folder named, "high-dim" on your Desktop which will store all of the code we generate throughout the workshop

- `C:\Users\username\Desktop\high-dim # Windows path`

- `/Users/username/Desktop/high-dim # Mac`

- `/home/username/Desktop/high-dim # Linux`
2. Download the helper_functions.py script and add it to the high-dim folder you just created.
Click to download: [helper_functions.py](/code/helper_functions.py)
{% include links.md %}



## Installing Python using Anaconda

[Python](https://python.org/) is a popular language for scientific computing, and a frequent choice
for machine learning as well. Installing all of its scientific packages
individually can be a bit difficult, however, so we recommend the installer [Anaconda](https://www.anaconda.com/products/individual)
which includes most (but not all) of the software you will need.

Regardless of how you choose to install it, please make sure you install Python
version 3.11. Also, please set up your python environment at
least a day in advance of the workshop.  If you encounter problems with the
installation procedure, ask your workshop organizers via e-mail for assistance so
you are ready to go as soon as the workshop begins.

### Windows - [Video tutorial](https://www.youtube.com/watch?v=xxQ0mzZ8UvA)

1. Open [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
   with your web browser.

2. Download the Python 3 installer for Windows.

3. Double-click the executable and install Python 3 using _MOST_ of the
   default settings. The only exception is to check the
   **Make Anaconda the default Python** option.

### Mac OS X - [Video tutorial][video-mac]

1. Open [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
   with your web browser.

2. Download the Python 3 installer for OS X.

3. Install Python 3 using all of the defaults for installation.

### Linux

Note that the following installation steps require you to work from the shell.
If you run into any difficulties, please request help before the workshop begins.

1.  Open [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution) with your web browser.

2.  Download the Python 3 installer for Linux.

3.  Install Python 3 using all of the defaults for installation.

    a.  Open a terminal window.

    b.  Navigate to the folder where you downloaded the installer

    c.  Type

    ~~~
    $ bash Anaconda3-
    ~~~
    {: .language-bash}

    and press tab.  The name of the file you just downloaded should appear.

    d.  Press enter.

    e.  Follow the text-only prompts.  When the license agreement appears (a colon
        will be present at the bottom of the screen) hold the down arrow until the
        bottom of the text. Type `yes` and press enter to approve the license. Press
        enter again to approve the default location for the files. Type `yes` and
        press enter to prepend Anaconda to your `PATH` (this makes the Anaconda
        distribution the default Python).

## Installing the required packages

[Conda](https://docs.conda.io/projects/conda/en/latest/) is the package management system associated with [Anaconda](https://anaconda.org) and runs on Windows, macOS and Linux.
Conda should already be available in your system once you installed Anaconda successfully. Conda thus works regardless of the operating system.
Make sure you have an up-to-date version of Conda running.
See [these instructions](https://docs.anaconda.com/anaconda/install/update-version/) for updating Conda if required.
{: .callout}

To create a conda environment called `highdim_workshop` with the required packages, open a terminal (Mac) or Anaconda prompt (Windows) and type the command:
~~~
conda create --name highdim_workshop python jupyter seaborn scikit-learn pandas statsmodels 
~~~
{: .source}

Activate the newly created environment:
~~~
conda activate highdim_workshop
~~~
{: .source}

## Starting Jupyter Lab

We will teach using Python in [Jupyter lab](http://jupyter.org/), a
programming environment that runs in a web browser. Jupyter requires a reasonably
up-to-date browser, preferably a current version of Chrome, Safari, or Firefox
(note that Internet Explorer version 9 and below are *not* supported). If you
installed Python using Anaconda, Jupyter should already be on your system. If
you did not use Anaconda, use the Python package manager pip
(see the [Jupyter website](http://jupyter.readthedocs.io/en/latest/install.html#optional-for-experienced-python-developers-installing-jupyter-with-pip) for details.)

To start jupyter lab, open a terminal and type the command:

~~~
$ jupyter lab
~~~
{: .source}

## Check your setup
To check whether all packages installed correctly, start a jupyter notebook in jupyter lab as
explained above. Run the following lines of code:
~~~
import sklearn
print('sklearn version: ', sklearn.__version__)

import seaborn
print('seaborn version: ', seaborn.__version__)

import pandas
print('pandas version: ', pandas.__version__)
~~~
{:.language-python}

This should output the versions of all required packages without giving errors.
Most versions will work fine with this lesson, but:
- For sklearn, the minimum version is 1.2.2

## Fallback option: cloud environment
If a local installation does not work for you, it is also possible to run this lesson using [Google colab](https://colab.research.google.com/). If you open a jupyter notebook from colab, the required packages are already pre-installed. Note that colab uses jupyter notebook instead of jupyter lab.
