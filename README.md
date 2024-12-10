---
title: "Predicting Food Insecurity Among Iowa Senior Populations"
author: "Trevor Stevens, Sophie Wokatsch, and Clayton Gustafson"
date: "12/09/2024"
output: html_document
---

## Introduction
This repository contains the code and data required to reproduce the predictive model and findings presented to our client, WesleyLife, at the conclusion of our Fall 2024 STAT 172 course taught by Dr. Lendie Follett at Drake University. The end models predict different measures of food insecurity among seniors by Public Use Microdata Area (PUMA) across the state of Iowa. For the purposes of this project, "seniors" are defined as individuals over 60 years of age. The "model_run_out" predicts the proportion of households containing seniors that reported "being worried that food would run out before being able to afford more during the past year". The "model_not_balanced" predicts the proportion of households containing seniors that reported they "could not afford to eat balanced meals in the past year".

## Requirements
To install the required R packages, run the following code in R:


```r
install.packages(c("tidyverse", "ggthemes", "logistf", "glmnet", "haven",
                   "randomForest", "rpart", "rpart.plot", "pROC", "reshape2",
                    "knitr"))
```

## Data

We use three sources of data in our project...

1. CPS data - Collected at the individual level with additional household information. Contains food insecurity measures.
We utilized this data to train our model and decide which variables to include in our final models.

2. ACS data - Collected at the individual level with additional household information, but also contains
PUMA. Does not contain food insecurity measures. We utilized this data to test our final models.

3. Total Iowa Seniors by PUMA - Contains the number of seniors in each Iowa PUMA. Appended to final model outputs to calculate
the predicted number of food insecure seniors in each PUMA.

The data files that will be called are "cps_00005.csv" (CPS data), "spm_pu_2022.sas7bdat" (ACS data), and 
"total_iowa_seniors_by_puma.csv" (Total Iowa Seniors by PUMA).

The CPS data and Total Iowa Seniors by PUMA data are contained in this repository. The ACS data can be downloaded from the
United States Census website.

## Reproduce
1. Download `clean_cps.R` and `clean_acs.R` to ensure that code will run that sources these R scripts. These two scripts help
clean the CPS data and the ACS data respectively, ensuring that each dataset has the same variable names, all unneccessary variables
are removed, and weights are added. 
  
2. Run `model_run_out.R` to reproduce exploratory random forests, trees, and clustering. The code also reproduces
all versions of the final model that were tested and outputs visualizations that were presented to WesleyLife.
It is notated in the code which model was the final selection for easy further exploration.  

3. Run `model_not_balanced.R` to reproduce exploratory random forests, trees, and clustering. The code also reproduces
all versions of the final model that were tested and outputs visualizations that were presented to WesleyLife.
It is notated in the code which model was the final selection for easy further exploration.

## Explanation of Methods
1. Selection of explanatory variables: Demographic characteristics (gender, race, education, marital status) were represented
as a proportion of the total household. Other household characteristics (household size, number of kids, number of seniors)
were represented as counts. This is because for demographic characteristics, the meaning lies in the makeup of the household,
not the count of a particular gender or race. For the other characteristics, we believed that an increase in their total
number would have greater predictive influence on food insecurity levels.

2. Selection of final models: Lasso and Ridge regression models were tested to model our response variables using our
explanatory variables. They were tested with only the explanatory variables, with 5 different clusters calculated in our
model files, and with both the explanatory variables and clusters. We selected the Ridge regression model using only
the explanatory variables for both model_run_out and model_not_balanced. This is because it produced a high AUC
and because of its interpretability.

3. Investigation of education_prop for run_out: Further analysis was done on education levels for the run_out food
insecurity measure. This is because education_prop was the highest node on the decision tree for this model,
making it one of the most important predictors in our model.

4. Investigation of married_prop for not_balanced: Further analysis was done on marital status for the not_balanced food
insecurity measure. This is because married_prop was the highest node on the decision tree for this model,
making it one of the most important predictors in our model.
