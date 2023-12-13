# Exploring Stock Graphs for Stock Price Prediction

This is my individual final project for the CPSC 483/583 course *Deep Learning on Graph structure Data* at Yale University at fall AY23/24. Please take a look at my final report to learn the models I used.

I want to express my gratitude to Prof. Ying and all the TAs of this course. The course is well-organized and comprehensive. I learnt a lot of great ideas and techniques in deep learning in this course.

I love the idea of message passing :)

## Environment
The experiment is run with python 3.8. Download our requirements.txt file and run the following code.
```{bash}
pip install -r requirements.txt
```

## Datasets
The datasets I used are forked from https://github.com/fulifeng/Temporal_Relational_Stock_Ranking. 

## Run Experiment
The results mentioned in the report can be reproduced by running the following code. At this stage, we do not add any arguments for the script. In addition, the model training can be super slow (~24hr on A100...be patient). The training speed may be further optimized by split the large graph into small pieces since one of our stock graphs is blockwise fully connected. 

```{bash}
python main.py
```
