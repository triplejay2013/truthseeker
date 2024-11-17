For this project we are going to explore the Truthseeker dataset.  You may work on teams of up to 4 people.  

Your goal is to achieve the best possible performance as measured by classification accuracy on an 80/20 train/test split of the data.  You should test on both the 2-class and 4-class versions of this problem.  Your should split the data grouped by the statement.  i.e. no example in the validation set should share a statement with an example in the training set. 

Graduate student teams should also explore the problem of labelling statements as either true or false based on the tweets.  In this case, you can post-process the dataset so that each statement is associated with all of the tweets about it.  i.e. you try to determine if a statement is false based on all the tweets that have been made about it.

You may use whatever source code resources (Jupyter notebook, python code, etc) you can find on the internet, but please make sure you reference where you found them and clearly discuss how you used them.  

You may use either the dataset for traditional ML techniques, or the dataset for use with DNNs like BERT.  Or both.

Be prepared to run your best algorithm on any new data I send you (we will do our best to get some new data from twitter, hopefully).

You are to submit a writeup and the link to your GitHub or gitlab repository that contains your source code.   As with the previous assignment, You must use either the LaTex. or (if you have to) Word. template located here. to write your paper (this is the journal format for the IEEE Transactions on Pattern Analysis and Machine Intelligence). 

I expect you to expend some effort to try to optimize your performance.  Your writeup should document this effort.  Your writeup should also carefully describe your final approach, your experimental setup, have well-conceived figures that present your data, with descriptive captions, and provide thoughtful discussion and analysis.  

I have tried to link the data below.  I have also put the data on Theia in a Truthseeker directory.

Features_For_Traditional_ML_Techniques.csvDownload Features_For_Traditional_ML_Techniques.csv

readme.txt Download readme.txtreadme.txt Download readme.txt

Truth_Seeker_Model_Dataset_With_TimeStamps 1.xlsxDownload Truth_Seeker_Model_Dataset_With_TimeStamps 1.xlsx

Truth_Seeker_Model_Dataset.csv Download Truth_Seeker_Model_Dataset.csv 

 

Here's a notebook that Casey developed to make it easy to fine tune a BERT-based classification model:

llm-finetune-new.ipynb.txtDownload llm-finetune-new.ipynb.txt

It's meant to be generic so if you can load the data and put it into two lists for X inputs and y labels, then it kind of does the rest. You have have to change the loss function depending on your data. 

 

Here's a repository I found on GitHub focused on the truth seeker dataset:

https://github.com/akash3patel/Truthseeker/tree/mainLinks to an external site.

Here's a Jupyter notebook we'll talk about in class: Analysis_v2.ipynb