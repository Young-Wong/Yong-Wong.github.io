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

