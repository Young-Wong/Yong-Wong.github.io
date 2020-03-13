---
layout: post
title: LendingClub Fraud Detection Project
date: 2020-03-12
categories: xgboost
tags: Project Xgboost Fintech
---

## Background
 
Lending club is one of the largest peer-to-peer lending company in the United State and also the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission, and to offer loan trading on a secondary market (https://en.wikipedia.org/wiki/LendingClub). LendingClub offers the borrowing and investing solution for individual investors with personal loans, auto refinancing loans. Business loans and medical financing loans. In the meantime, the default risk exists among the different kinds of p2p loans which prevents the investor from getting their money back or increasing their investment amount. 

## Task Purpose

This project focuses on analyzing the features of the every loans in the history of LendingClub and define the most import features that would most likely have impact on the default risk. Our goal is to build a model to learn the data provided by LendingClub and predict the risk of default for the future loans.
     
## Metric and Evaluation

The metric we are using to evaluate the machine learning model is AUC, area under the ROC curve between the predicted probability and the observed target. For example for each Loan ID in the dataset, we will predict a probability of the default. 
 
## Data

We got the historical data from LendingClub csv file and the current via API https://api.lendingclub.com/api/investor/v1/loans/listing like below:

![](assets/img/fintech/img01.png)

## EDA
 
After loading the data, we found out that the dataset has 89 numeric features but 29 of them are all Nulls and 19 categorical features.    

![](assets/img/fintech/img02.png)
![](assets/img/fintech/img03.png)

## Insights and Plots
 
The feature installment is the amount of monthly payment for each loan. By plotting a histogram graph on installment, we can see that the monthly payment of $200 - $400 has the highest counts among all the installment amount. This applies that people tend to accept the monthly installment around $200 to $400. 

![](assets/img/fintech/img04.png)

In terms of analyzing whether a loan can be default, the borrower's income verification status is a good aspect. Usually if the borrower's source of income is verified, then the risk of the default is lower and vice versa. However, we drew a graph on the feature named verificationstatus and the result doesn’t look like what we imaged. According to the graph, the counts of default loan that have an income verified has a very high rate around 20%. This is a very interesting open question for us to think about. 

![](assets/img/fintech/img05.png)

Another important feature to consider is what type of home ownership did the borrower have.  We also plotted the bar charts of the default counts by homeownership. According to the graph, borrowers who don’t own their house and they pay rent to live have a higher risk of default rate. What surprised us is that people who are paying mortgage is less likely to be default than people own their home without any mortgage. 

![](assets/img/fintech/img06.png)


## Features
 
For an issued loan, we found out that the column fundedamnt is equal to loanamnt. Since we will only do analysis on issued loan, we can drop fundedamnt for our task purpose. 

![](assets/img/fintech/img07.png)

We did some feature engineering on the data time when the borrowers' earliest reported credit line was opened. Usually people who open their credit earlier means they have been in the credit system for a longer time and less likely to have default loan. SO we have changed the date time of the column earliestcrline into numeric number.

![](assets/img/fintech/img08.png)
![](assets/img/fintech/img09.png)
![](assets/img/fintech/img10.png)


For the rest part of the feature engineering, we have converted the interest rate and the other rate by percentage data type into numeric. And we did the ordinal feature encoding for grade (LC assigned loan grade) and subgrade. We have also did the one-hot-encoding of some categorical features like homeownership, verificationstatus, purpose and initialliststatus in order to fit into the model like Xgboost for future model training (figure shows above).

## Model
 
By going through the process of EDA, we gained some helpful insights on the context of the dataset. And we have also specifically defined our goal of the model is to take and analysis the data with term =36 months and build a model to train the data using the October to December dataset and test the model using the rest of the data.
 
## Description

The centerpiece of model’s entire operation is using algorithm to choose which notes to invest in. Often investors have to choose between hundreds or thousands of available loans at Lending Club. This model makes the process easier by using machine-learning to calculate which notes are more likely to perform better than others. The moment new loans are added to the platforms, the algorithm analyzes the variables of these loans and only invests in the best ones. The entire process, again, takes a split second.
  
## Pros and Cons

The model we finally decided to use is Xgboost. Xgboost is famous for high accuracy and great performance. Since our dataset doesn't have a very huge volume, Xgboost will give us a great result and also it take too much time on training. The algorithm that XgBoost used is Gradient Boosted Regression Trees (GBRT). The advantages of GBRT are: Natural handling of data of mixed type (= heterogeneous features), Predictive power, Robustness to outliers in output space (via robust loss functions).
 
## Result
 
After several manually model tuning, we came up with a model that has the highest performance. The result we generated from this model based on AUC/ROC metric is :
 
Area under the ROC curve - validation: 0.699226
Area under the ROC curve - train: 0.802750
Area under the ROC curve - test: 0.710706

![](assets/img/fintech/img11.png)
![](assets/img/fintech/img12.png)

## Conclusion
 
Based on the train model, we have also checked and generated the ranking of the features according to the importance of the default risk. The top feature is called dti which is a ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income. The second top feature is mo_sin_old_il_acct which is Months since oldest bank installment account opened. This means the longer the borrowers exist in the credit system, the less likely they default their loan. There are other important features such as Annual Income, Total Installment high credit/credit Limit, Revolving Credit Limit and Portion of Balances.

![](assets/img/fintech/img13.png)
 
 
## Model distribution
 
We have saved our model into pkl file. And distributed our trained model into the python Flask system. Our team have built a user friendly front end website page using HTML CSS and python Flask to display our result and prediction. The UI looks like below. As you can see, five key features was picked according to the feature-importance scores: Debt-to-Income Ratio, Annual Income, Total Installment high credit/credit Limit, Revolving Credit Limit, Portion of Balances.
 
This model makes the process easier by using machine-learning to calculate which notes are more likely to perform better than others. The moment new loans are added to the platforms, the algorithm analyzes the variables of these loans and only invests in the best ones. The entire process, again, takes a split second.

![](assets/img/fintech/img14.png)




























