source('code/clean_cps.R')
source('code/clean_acs.R')

not_balanced_data <- cps_data %>% drop_na(not_balanced)

### CLASSIFICATION TREE ####

library(rpart)
library(rpart.plot)
library(pROC)

# split data into train and test 

RNGkind(sample.kind="default")
set.seed(1981536)

train.idx <- sample( x= c(1:nrow(not_balanced_data)), size=.8*nrow(not_balanced_data))

# make training dataframe
train.df <- not_balanced_data[train.idx,]

# make a mutually exclusive testing dataframe
test.df <- not_balanced_data[-train.idx,]

# grow a very large tree
set.seed(172172172)
ctree <- rpart(not_balanced ~ female_prop + hispanic_prop + black_prop
               + kids_count + elderly_count + education_prop
               + married_prop,
               data = train.df ,
               method = "class",
               control = rpart.control(cp=0.0001, minsplit = 1 ))
printcp(ctree)

# again, the tree with the optimal CP here would be ridiculous
# lets try again, but set a maximum argument:

ctree2 <- rpart(not_balanced ~ female_prop + hispanic_prop + black_prop
                + kids_count + elderly_count + education_prop
                + married_prop,
                data = train.df ,
                method = "class",
                control = rpart.control(cp=0.0001, minsplit = 1,maxdepth = 5 ))
printcp(ctree2)


# this tree is a reasonable size and not overfit like it would be if we used
# a different stopping rule, so lets use this one.

# lets make some observations based on the tree:

rpart.plot(ctree2)

# at the top of the tree, we are split by married_prop. This is the proportion
# of those in the household that are married. Because it is the first split,
# we know that this is an important variable. The next two most important variables,
# based on their order of appearance in the tree, are female proportion and kids count,
# followed by elderly count. 



# let's make some predictions:

pi_hat <- predict(ctree2,test.df, type ="prob")[,"1"]

# make ROC curve
rocCurve <- roc(response=test.df$not_balanced, 
                predictor = pi_hat, 
                levels = c("0","1"))

plot(rocCurve, print.thres=TRUE,print.auc=TRUE)
# the pi star, aka the optimal cutoff point is 0.349.
# this means that we predict a "1" (do not have enough / enough kinds of food 
# in the household) if the predicted probability is > 0.349, and we predict
# a "0" (feels that they do have enough) if the predicted probability is 
# less than 0.349. 
# The specificity is 0.750, which means that we have a probability of 0.750 of true
# negatives, which means we predict they have enough and enough kinds of foods when
# they actually do. (true negatives, meaning that we predict food insecurity
# is not present, and it actually isn't)
# the sensitivity is 0.446, which means that we have a probability of 0.481 of
# true positives, which means that we predict they do not have enough kinds of foods
# in the house, and they actually do not have enough kinds of foods. 

# The AUC is 0.614, which is not that high, leading me to think a tree may not
# be the best predictor. 

# However, we were able to learn about what some of the most important variables
# may be for predicting households that do not have enough and enough kinds of
# foods in the household. These are, in order, married proportion, female proportion
# & kids count, and elderly count.

### FOREST ###

# split data into train and test 

RNGkind(sample.kind="default")
set.seed(1981536)

train.idx <- sample( x= c(1:nrow(not_balanced_data)), size=.8*nrow(not_balanced_data))

# make training dataframe
train.df <- not_balanced_data[train.idx,]

# make a mutually exclusive testing dataframe
test.df <- not_balanced_data[-train.idx,]

set.seed(172172172)

# Baseline forest
base_forest <- randomForest(not_balanced ~ female_prop + hispanic_prop + black_prop
                            + kids_count + elderly_count + education_prop
                            + married_prop,
                            data = train.df,
                            ntree = 1000,
                            weights = as.integer(train.df$weight),
                            mtry = 3, # sprt(7) = 2.646
                            importance = TRUE)
base_forest

# Tuning
mtry <- c(1:7) # 7 variables used as X's

# Make room for m, OOB error
keeps2 <- data.frame(m = rep(NA, length(mtry)),
                     OOB_err_rate = rep(NA, length(mtry)))

for (idx in 1:length(mtry)){
  tempforest <- randomForest(not_balanced ~ female_prop + hispanic_prop + black_prop
                             + kids_count + elderly_count + education_prop
                             + married_prop,
                             data = train.df,
                             ntree = 1000,
                             weights = as.integer(train.df$weight),
                             mtry = mtry[idx])
  
  keeps2[idx, 'm'] <- mtry[idx]
  
  keeps2[idx, 'OOB_err_rate'] <- mean(predict(tempforest)!= train.df$not_balanced)
  
}
keeps2

ggplot(data = keeps2) +
  geom_line(aes(x=m, y=OOB_err_rate)) +
  theme_bw() + labs(x = 'm (mtry) value', y = 'OOB error rate')

# Best OOB had mtry = 4

final_forest <- randomForest(not_balanced ~ female_prop + hispanic_prop + black_prop
                             + kids_count + elderly_count + education_prop
                             + married_prop,
                             data=train.df,
                             ntree=1000,
                             weights=as.integer(train.df$weight),
                             mtry=4) # Based on tuning

### ROC Curve ###

pi_hat <- predict(final_forest, test.df, type = "prob")[,1] # POSITIVE EVENT = 1!
rocCurve <- roc(response = test.df$not_balanced, # Give it truth
                predictor = pi_hat, # Probabilities of positive event 1
                levels = c(0,1)) 

plot(rocCurve, print.thres = TRUE, print.auc = TRUE)

# Pi * = 0.911
# AUC = 0.757
# Specificity = 0.750 (first number in graph)
# Sensitivity = 0.655


##### Lasso and Ridge Regression #####

#Make a training and testing DATA FRAME
RNGkind(sample.kind = "default")
set.seed(23591)
train.idx = sample(x = 1:nrow(not_balanced_data), size = .7*nrow(not_balanced_data))
train.df = not_balanced_data[train.idx,]
test.df = not_balanced_data[-train.idx,]

#Make a training and testing DATA MATRIX
x.train = model.matrix(not_balanced ~ hhsize + female_prop + hispanic_prop + black_prop
                       + kids_count + elderly_count + education_prop + 
                         married_prop, data = train.df)[,-1] 
x.test = model.matrix(not_balanced ~ hhsize + female_prop + hispanic_prop + black_prop
                      + kids_count + elderly_count + education_prop + 
                        married_prop, data = test.df)[,-1]

#Make VECTORS of run_out (our Y variable)
y.train = as.vector(train.df$not_balanced)
y.test = as.vector(test.df$not_balanced)

#Use cross validation to fit (LOTS OF) lasso and ridge regressions
lasso_cv_nb = cv.glmnet(x.train,
                        y.train,
                        family = binomial(link = "logit"),
                        alpha = 1,
                        weights = as.integer(train.df$weight))

ridge_cv_nb = cv.glmnet(x.train,
                        y.train,
                        family = binomial(link = "logit"),
                        alpha = 0,
                        weights = as.integer(train.df$weight))

#Choose the lambda value that minimizes out of sample error
best_lasso_lambda = lasso_cv_nb$lambda.min
best_ridge_lambda = ridge_cv_nb$lambda.min

#Final Lasso and Ridge models
final_lasso_nb = glmnet(x.train, y.train, 
                        family = binomial(link = "logit"),
                        alpha = 1,
                        lambda = best_lasso_lambda,
                        weights = as.integer(train.df$weight))
final_ridge_nb = glmnet(x.train, y.train, 
                        family = binomial(link = "logit"),
                        alpha = 0,
                        lambda = best_ridge_lambda,
                        weights = as.integer(train.df$weight))

##### Testing Model Performance #####

#QUANTIFY PREDICTION PERFORMANCE OF ALL 3 MODELS
test.df.preds = test.df %>% 
  mutate(
    lasso_pred_nb = predict(final_lasso_nb, x.test, type = "response")[,1],
    ridge_pred_nb = predict(final_ridge_nb, x.test, type = "response")[,1]
  )

#FIT ROC CURVES

lasso_rocCurve_nb = roc(response = as.factor(test.df.preds$not_balanced), 
                        predictor = test.df.preds$lasso_pred_nb, 
                        levels = c("0", "1")) 

ridge_rocCurve_nb = roc(response = as.factor(test.df.preds$not_balanced), 
                        predictor = test.df.preds$ridge_pred_nb, 
                        levels = c("0", "1")) 

plot(lasso_rocCurve_nb, print.thres = TRUE, print.auc = TRUE) # AUC = 0.621
plot(ridge_rocCurve_nb, print.thres = TRUE, print.auc = TRUE) # AUC = 0.622



#### CLUSTERING TO PREDICT ####

# stack cps and acs to cluster
summary(acs_data)

# Reorder / keep only same columns
acs_df <- acs_data[, c("hhsize", "female_prop","hispanic_prop","black_prop",
                       "kids_count","elderly_count","education_prop"
                       ,"married_prop")]

summary(cps_data)

cps_df <- cps_data[, c("hhsize", "female_prop","hispanic_prop","black_prop",
                       "kids_count","elderly_count","education_prop"
                       ,"married_prop")]

# make sure they are the same
summary(cps_df)
summary(acs_df)

stacked_data <- rbind(cps_df, acs_df)

summary(stacked_data)

# perform clustering

library(reshape2) #For melting data frame

#Drop all NA values since we cannot have them in the data for clustering
stacked_data_X = drop_na(subset(stacked_data, select = c(female_prop, hispanic_prop,
                                                         black_prop, kids_count,
                                                         elderly_count, education_prop,
                                                         married_prop, hhsize)))

#Standardize the columns
stacked_data_stand = apply(stacked_data_X, 2, function(x){(x - mean(x))/sd(x)})

#Compute observation-observation distances
stacked_data_dist = dist(stacked_data_stand, method = "euclidean")

#Measure cluster-to-cluster similarity
stacked_data_clust = hclust(stacked_data_dist, method = "ward.D2")

#Making sense of the clusters and saving them
stacked_data$h_cluster = as.factor(cutree(stacked_data_clust, k=5))
stacked_data_X_long = melt(stacked_data, id.vars = c("h_cluster"))
head(stacked_data_X_long)
ggplot(data = stacked_data_X_long) +
  geom_boxplot(aes(x = h_cluster, y = value, fill = h_cluster)) +
  facet_wrap(~variable, scales = "free") +
  scale_fill_brewer("Cluster \nMembership", palette = "Dark2") +
  ggtitle("Hierarchical Clusters")

# Name clusters:
# 1: Female Small household elderly
# 2: Black Mixed family (elderly + kids)
# 3: Hispanic
# 4: Kids
# 5: White elderly

### ANALYZE CLUSTERS #####

# assign the clusters as a column in the data
stacked_data_clustered <- stacked_data
stacked_data_clustered$cluster <- as.factor(stacked_data$h_cluster)

# split the datasets back up

head(stacked_data_clustered)
head(cps_df)
# as we can see cps is "on top" in the stack.

# verify that the sum of the rows in both equals the rows in the stacked data
nrow(cps_df)
nrow(acs_df)
nrow(stacked_data_clustered)
19447+13721
# Yup!

# actually split them back up
cps_df_cluster <- stacked_data_clustered[1:nrow(cps_df), ]
acs_df_cluster <- stacked_data_clustered[(nrow(acs_df) + 1):nrow(stacked_data_clustered), ]

head(cps_df_cluster)
# the head is right! 
nrow(cps_df_cluster)
# and the number of rows is right.

# add the cluster column onto the original dataset with the Y variable info
cps_data$cluster <- cps_df_cluster$cluster
head(cps_data)


# find mean of not_balanced in each cluster
means <- cps_data %>%
  group_by(cluster) %>%
  summarize(Mean_not_balanced = mean(not_balanced, na.rm = TRUE))

means
# according to this cluster 2 has the highest concentration of people who
# feel as though they dont eat enough balanced meals in the past year.
# in order from most food insecurity to least food insecurity,
# cluster 2, 1, 3, 4, 5

#### CREATE MLE, LASSO and RIDGE MODELS FOR CLUSTERS ####

# remove NA's for the Y variable
not_balanced_data <- cps_data %>% drop_na(not_balanced)

RNGkind(sample.kind = "default")
set.seed(23591)


# create train and test dataset
train.idx.nb <- sample( x= c(1:nrow(not_balanced_data)), size=.8*nrow(not_balanced_data))

# make training dataframe
train.df.nb <- not_balanced_data[train.idx.nb,]

# make mutually exclusive testing dataframe
test.df.nb <- not_balanced_data[-train.idx.nb,]

#Make a training and testing DATA MATRIX
x.train.cluster = model.matrix(not_balanced ~ cluster, data = train.df.nb)[,-1] 
x.test.cluster = model.matrix(not_balanced ~ cluster, data = test.df.nb)[,-1]

#Make VECTORS of run_out (our Y variable)
y.train.cluster = as.vector(train.df.nb$not_balanced)
y.test.cluster = as.vector(test.df.nb$not_balanced)


#Use cross validation to fit (LOTS OF) lasso and ridge regressions
lasso_cluster_nb = cv.glmnet(x.train.cluster,
                             y.train.cluster,
                             family = binomial(link = "logit"),
                             alpha = 1,
                             weights = as.integer(train.df.nb$weight))

ridge_cluster_nb = cv.glmnet(x.train.cluster,
                             y.train.cluster,
                             family = binomial(link = "logit"),
                             alpha = 0,
                             weights = as.integer(train.df.nb$weight))


#Choose the lambda value that minimizes out of sample error
cluster_lasso_lambda = lasso_cluster_nb$lambda.min
cluster_ridge_lambda = ridge_cluster_nb$lambda.min


#Final Lasso and Ridge models
final_clust_lasso_nb = glmnet(x.train.cluster, y.train.cluster, 
                              family = binomial(link = "logit"),
                              alpha = 1,
                              lambda = cluster_lasso_lambda,
                              weights = as.integer(train.df.nb$weight))


final_clust_ridge_nb = glmnet(x.train.cluster, y.train.cluster, 
                              family = binomial(link = "logit"),
                              alpha = 0,
                              lambda = cluster_ridge_lambda,
                              weights = as.integer(train.df.nb$weight))


##### Testing cluster Model Performance #####

#QUANTIFY PREDICTION PERFORMANCE OF BOTH MODELS
test.df.preds = test.df.nb %>% 
  mutate(
    lasso_clust_pred_nb = predict(final_clust_lasso_nb, x.test.cluster, type = "response")[,1],
    ridge_clust_pred_nb = predict(final_clust_ridge_nb, x.test.cluster, type = "response")[,1]
  )

#FIT ROC CURVES

lasso_clust_roc_nb = roc(response = as.factor(test.df.preds$not_balanced), 
                         predictor = test.df.preds$lasso_clust_pred_nb, 
                         levels = c("0", "1")) 

ridge_clust_roc_nb = roc(response = as.factor(test.df.preds$not_balanced), 
                         predictor = test.df.preds$ridge_clust_pred_nb, 
                         levels = c("0", "1")) 

plot(lasso_clust_roc_nb, print.thres = TRUE, print.auc = TRUE) 
plot(ridge_clust_roc_nb, print.thres = TRUE, print.auc = TRUE)
# for the lasso, AUC is 0.570
# for the ridge, AUC is 0.571
# These models are even worse than the trees and the forest. 



#### MAKE LASSO AND RIDGE MODELS USING BOTH CLUSTERS AND X VARS ####

# I'm also including two interaction variables: married*education
# because in synergy, these represent somebody who is stereotypically well off.
# and elderly*kids, because these represent people who are likely unable to work
# but are still mouths to feed. In synergy, they show that it is a "mixed family"
# with multiple generations.

#Make a training and testing DATA MATRIX
x.train.all = model.matrix(not_balanced ~ cluster + hhsize + female_prop + hispanic_prop + black_prop
                           + kids_count + elderly_count + education_prop + 
                             married_prop + married_prop*education_prop +
                             elderly_count*kids_count ,
                           data = train.df.nb)[,-1]


x.test.all = model.matrix(not_balanced ~ cluster+ hhsize + female_prop + hispanic_prop + black_prop
                          + kids_count + elderly_count + education_prop + 
                            married_prop + married_prop*education_prop +
                            elderly_count*kids_count ,
                          data = test.df.nb)[,-1]


#Make VECTORS of run_out (our Y variable)
y.train.all = as.vector(train.df.nb$not_balanced)
y.test.all = as.vector(test.df.nb$not_balanced)

#Use cross validation to fit (LOTS OF) lasso and ridge regressions
lasso_all_nb = cv.glmnet(x.train.all,
                         y.train.all,
                         family = binomial(link = "logit"),
                         alpha = 1,
                         weights = as.integer(train.df.nb$weight))

ridge_all_nb = cv.glmnet(x.train.all,
                         y.train.all,
                         family = binomial(link = "logit"),
                         alpha = 0,
                         weights = as.integer(train.df.nb$weight))

#Choose the lambda value that minimizes out of sample error
all_lasso_lambda = lasso_all_nb$lambda.min
all_ridge_lambda = ridge_all_nb$lambda.min


#Final Lasso and Ridge models
final_all_lasso_nb = glmnet(x.train.all, y.train.all, 
                            family = binomial(link = "logit"),
                            alpha = 1,
                            lambda = all_lasso_lambda,
                            weights = as.integer(train.df.nb$weight))


final_all_ridge_nb = glmnet(x.train.all, y.train.all, 
                            family = binomial(link = "logit"),
                            alpha = 0,
                            lambda = all_ridge_lambda,
                            weights = as.integer(train.df.nb$weight))


#QUANTIFY PREDICTION PERFORMANCE OF MODELS
test.df.preds = test.df.nb %>% 
  mutate(
    lasso_all_pred_nb = predict(final_all_lasso_nb, x.test.all, type = "response")[,1],
    ridge_all_pred_nb = predict(final_all_ridge_nb, x.test.all, type = "response")[,1]
  )

#FIT ROC CURVES

lasso_all_rocCurve_nb = roc(response = as.factor(test.df.preds$not_balanced), 
                            predictor = test.df.preds$lasso_all_pred_nb, 
                            levels = c("0", "1")) 

ridge_all_rocCurve_nb = roc(response = as.factor(test.df.preds$not_balanced), 
                            predictor = test.df.preds$ridge_all_pred_nb, 
                            levels = c("0", "1")) 

plot(lasso_all_rocCurve_nb, print.thres = TRUE, print.auc = TRUE) # AUC = 0.612
plot(ridge_all_rocCurve_nb, print.thres = TRUE, print.auc = TRUE) # AUC = 0.614
# these are also not the best numbers we could be seeing

##### Final Model Prediction and Analysis #####

final_nb_model = final_ridge_nb

#### PREDICT ####

summary(acs_data)

acs_matrix <- as.matrix(subset(acs_data, select = -c(serialno, PUMA, weight)))

acs_data$prediction <- predict(final_nb_model,acs_matrix,type="response")[,1]

summary(acs_data)


##### Aggregate ACS Data by PUMA #####
acs_data_sub <- acs_data %>% filter(elderly_count >=1)
#agg_ACS = aggregate(acs_data_sub$prediction,
#                    by = list(acs_data_sub$PUMA),
#                    FUN = weighted.mean(w = acs_data_sub$weight))
#the above may not work to take a weighted average

acs_data_sub_agg <- acs_data_sub %>% 
  group_by(PUMA) %>% 
  summarise(mean_not_balanced = weighted.mean(x = prediction, w = weight),
            mean_education = weighted.mean(x = education_prop, w = weight),
            mean_hhsize = weighted.mean(x = hhsize, w = weight),
            mean_female = weighted.mean(x = female_prop, w = weight),
            mean_hispanic = weighted.mean(x = hispanic_prop, w = weight),
            mean_black = weighted.mean(x = black_prop, w = weight),
            mean_kids = weighted.mean(x = kids_count, w = weight),
            mean_seniors = weighted.mean(x = elderly_count, w = weight),
            mean_married = weighted.mean(x = married_prop, w = weight))


###### Find Number of Food Insecure Seniors in Each PUMA #####

#Read in the data with number of seniors in each PUMA
PUMA_data = read.csv('data/total_iowa_seniors_by_puma.csv')
summary(PUMA_data)
str(PUMA_data)

#Merge the number of seniors in each PUMA to the aggregated ACS data
merged_ACS = merge(acs_data_sub_agg, PUMA_data, by.x = "PUMA", by.y = "GEOID")
summary(merged_ACS)

#Make a column with the number of food insecure seniors in each PUMA

merged_ACS <- merged_ACS %>% 
  mutate(insecure_seniors = mean_not_balanced*senior_population)

### VISUALIZATIONS ###

# Scatterplot for not_balanced married
ggplot(merged_ACS, aes(x=mean_married, y=mean_not_balanced)) +
  geom_point(size=3) +
  labs(
    title="Effect of Marital Status on Ability to Afford Balanced Meals",
    x = "Average Proportion of Married Individuals per Household",
    y = "Proportion of Households Unable to Afford Balanced Meals"
  ) +
  geom_smooth(method="lm", se=FALSE, color = "red") +
  theme_bw()

# PUMA 1901503 has the highest percent of citizens who don't get balanced meals,
# which is Des Moines City

# Trend Line Coefficient
coefficients(lm(mean_not_balanced ~ mean_married, data = merged_ACS))

# -0.1737165

### Tables ###
sorted_data <- merged_ACS %>% arrange(desc(insecure_seniors))
names(sorted_data)

library(knitr)
kable(sorted_data[, c("PUMA","insecure_seniors","senior_population", "mean_married")] %>% slice(1:5))


