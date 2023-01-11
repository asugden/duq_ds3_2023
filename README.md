# duq_ds3_2023

A repository of course materials for Duquesne University's Data Science 3 course, 2023.

In this class, we are a data science team with expectations to match. You are welcome to ask others for help, but at the same time are required to do your own work.

# Syllabus

We will have days that alternate theory with practice. Practice will involve both my coding in front of you, as well as taking turns coding in front of each other. Given that many job interviews require public coding, this is excellent, if occasionally challenging experience.

Note that we will adapt later weeks of the course to the interests of everyone in the course.

# Grading

It is easy to do well in this course. Seriously. Show up, engage, have fun, take notes, do the homework, and program occasionally on your own throughout the semester.

Because we are a data science team, treat this as you would a job. It matters that you show up and that you show interest. I'd much rather work with someone who is engaged than someone who plays on their phone.

Grades in this class are based on creating tools that are readable, runnable, and well-documented. 60% of the grade is code quality, 40% is documentation quality.

There will be a small number of short written assignments where brevity is required. The challenge is not length, but thoughtfulness.

Ample time will be given for homeworks, and they should be viewed both as a chance to practice and a mechanism of expanding the documented code in your Github (which will help with future jobs). Collaboration is highly valued, but if overly similar code is submitted, I reserve the right to request independent explanations of how the code works and the thought behind its organization. Push commits to Github frequently. If code appears to be duplicated and is committed in a single entry after the commits of matching code, the history of the code becomes clear.

If you are concerned about the final exam, you are welcome to meet with me to do a practice prior to the exam. Whether you've completed the homework and documented it well is obvious. But, if you are concerned with whether or not you are sufficiently participating, you can schedule time to chat about it at any point during the semester, and we can talk about mechanisms of improvement if necessary. Given these open channels of communication, however, requests to change a grade at the end of the semester will not go well. Assume that I am giving you the benefit of the doubt.

## Final exam

The final exam will be a mock job interview. It will be comprised of the following portions:

- Evaluation of Github, all of which will be filled in the previous weeks (is there high-quality Python code with functions, documentation, and a cohesive project): 40% (If you cared about your work during the semester and have put substantial effort into your Github, the weight of the coding challenge will decrease and the weight of the Github will increase.)
- Description of previous work: 40%
- Coding challenge: 20%

# Code necessities

I highly recommend using a unix-based system for programming-- either dual-boot a windows machine with Ubuntu or use a Mac. This computer lab is fantastic and has computers that can boot into Ubuntu. If necessary, then the easiest way to use Python on a Windows machine is using a combination of Anaconda and Git Bash.

To install Git bash: https://git-scm.com/download/win

(Given that Anaconda was used in DS1, I am assuming that you have already installed Anaconda.)

To ensure that you can use Anaconda from within Git Bash, checkout the first answer to this Stack Overflow question: https://stackoverflow.com/questions/54501167/anaconda-and-git-bash-in-windows-conda-command-not-found

It may seem weird that a course syllabus links to Stack Overflow, but you will find that the professional experience of Data Science often involves using Google for solutions. Installing core Unix components is one of the most frustrating tasks in computing, regardless of operating system, but doing so provides excellent experience.

## SSH Issues

For Github, if you want to use ssh, there are a variety of ways to deal with this.

1. PyCharm: https://www.jetbrains.com/help/pycharm/create-ssh-configurations.html
1. Github more generally: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

## Slack

You may have noticed that you've been sent a slack invite. This is never required, but it's an easy way to contact me if you have questions. It's also a great place to ask other folks in the class questions that you might have. There's nothing worse than getting stuck programming, and having others around is key.

## Notes

1. Set up your own Github account
1. Github accounts should include at least one example of Python, with docstrings
1. I recommend using VSCode + venv for installation
1. Let's use Python 3.8+

# Schedule

## Week 1

- Discussing each student's desired outcomes from the course
- VSCode/local coding
- Code documentation
  - Fuctions
  - Type hinting
  - Docstrings
  - Libraries versus scripts
  - Colab vs VSCode
- Github
  - Setup
  - READMEs
  - gitignore
  - cloning
  - adding
  - commiting
  - pushing
  - pulling
  - branching
  - diff

Evaluation:

- Set up Github account, add at least one repository with documentation
- Set up VSCode locally

## Week 2

Dataset: https://github.com/nytimes/covid-19-data/blob/master/us-counties.csv

Evaluation: https://www.youtube.com/watch?v=_uQrJ0TkZlc through Classes

- Pandas
  - All Pandas, all the time
  - Comfort and speed
  - Good habits such as apply
  - Bad habits such as for loops
  - Data merging
- Pandas is easy, efficiency is hard
- Numpy/Pandas translation

Evaluation:

- Identify interesting datasets
- Combine, subset, and clean datasets
- Give at least one bad example and one good example
- Convert from pandas to numpy and back

## Week 3-4

- Classification introduction
  - Examples of classification in data science
  - Discussion of classifier types
  - Formatting data for classification
  - Scikit-learn

Evaluation:

- Run a classifier
- Create a reusable function for training a classifier
- Create a reusable function for saving a classifier
- Create a reusable function for loading and running a classifier

## Week 5-7

- Decision trees
- Random forest
- XGBoost

Evaluation:

- Create an XGBoost classifier with class wrapper

## Remaining topics

- Classification

  - Bayesian
  - SVM
  - Catboost

- Classifier evaluation

  - Confusion matrix
  - F1 score, accuracy, precision, recall
  - Diagnostic plotting - single biggest differentiator
    - Histograms
    - Timeseries plots

- Bias in underlying datasets
- Bootstrapping + class mislabels

- Deep classification with neural nets

  - Fundamentals of neural networks
    - Relu
    - Sigmoid
    - Error functions
  - CNNs
  - LSTMs
  - Deep NNs

- Natural language processing
- Embedding
- Nearest neighbor search

- Databases

  - No SQL
  - SQL
  - ETL
  - Spark
  - Databricks

- AWS
- Google cloud

<!-- ## Week 5:

## Week 6:

## Week 7:

## Week 8:

## Week 9:

## Week 10:

## Week 11:

## Week 12:

## Week 13:

## Week 14:

## Week 15:-->

## Week 16

Final exam
