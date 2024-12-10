source('code/clean_cps.R')
source('code/clean_acs.R')

run_out_data <- cps_data %>% drop_na(run_out)



### CLASSIFICATION TREE ####
library(rpart)
library(rpart.plot)
library(pROC)

# split data into train and test 

RNGkind(sample.kind="default")
set.seed(1981536)

train.idx <- sample( x= c(1:nrow(run_out_data)), size=.8*nrow(run_out_data))

# make training dataframe
train.df <- run_out_data[train.idx,]

# make a mutually exclusive testing dataframe
test.df <- run_out_data[-train.idx,]

# grow a very large tree
set.seed(172172172)
ctree <- rpart(run_out ~  female_prop + hispanic_prop + black_prop
               + kids_count + elderly_count + education_prop
               + married_prop,
               data = train.df ,
               method = "class",
               control = rpart.control(cp=0.0001, minsplit = 1 ))
printcp(ctree)


# now, prune the tree
optimalcp <- ctree$cptable[which.min(ctree$cptable[,"xerror"]),"CP"]



# wow! according to the output, the best tree (smallest xerror) is at 260
# splits. That's kind of ridiculous to plot, so we can instead just look at
# some of the first splits to see what variables might be important.

ctree2 <- prune(ctree,cp=optimalcp)
ctree2

# in order of appearance, the 4 "most important" variables are 
# education proportion, kid count, black proportion, and elderly count


# lets try a different stopping rule to see if we would also get a huge tree
cutoff <- min(ctree$cptable[,"xerror"]) +
  ctree$cptable[which.min(ctree$cptable[,"xerror"]),"xstd"]
optcp <- ctree$cptable[which(ctree$cptable[,"xerror"]<cutoff)[1], "CP"]
ctree3<- prune(ctree, cp=optcp)
rpart.plot(ctree3)

# okay, it seems just as big this time.
# for simplicity, and to avoid overfitting, lets look at a smaller tree.

ctree4 <- rpart(run_out ~ female_prop + hispanic_prop + black_prop
                + kids_count + elderly_count + education_prop
                + married_prop,
                data = train.df ,
                method = "class",)
rpart.plot(ctree4)


# because this one is the only one of reasonable size, lets roll with it.

# let's make some predictions:

pi_hat <- predict(ctree4,test.df, type ="prob")[,"1"]

# make ROC curve
rocCurve <- roc(response=test.df$run_out, 
                predictor = pi_hat, 
                levels = c("0","1"))

plot(rocCurve, print.thres=TRUE,print.auc=TRUE)
# the pi star, aka the optimal cutoff point is 0.370.
# this means that we predict a "1" (feels that they will run out of food and
# be unable to buy more) if the predicted probability is > 0.370, and we predict
# a "0" (does not feel they will run out) if the predicted probability is 
# less than 0.370. 
# The specificity is 0.640, which means that we have a probability of 0.64 of 
# predicting not feeling they will run out of food when they actually feel they
# will not.
# the sensitivity is 0.548, which means that we have a probability of 0.548 of
# true positives, which means that we predict they feel like they will run out 
# of food when they do feel like they will run out of food and be unable to afford
# more. 
# The AUC is 0.610, which is not that high, leading me to think a tree may not
# be the best predictor. 

# However, we were able to learn about what some of the most important variables
# may be for predicting households that are worried they will run out of food
# without being able to afford more. These are: education proportion, black proportion
# and kids count, and elderly proportion (in order).

### FOREST ###

# split data into train and test 

RNGkind(sample.kind="default")
set.seed(1981536)

train.idx <- sample( x= c(1:nrow(run_out_data)), size=.8*nrow(run_out_data))

# make training dataframe
train.df <- run_out_data[train.idx,]

# make a mutually exclusive testing dataframe
test.df <- run_out_data[-train.idx,]

# grow a very large tree
set.seed(172172172)

# Baseline forest
base_forest <- randomForest(run_out ~ female_prop + hispanic_prop + black_prop
                            + kids_count + elderly_count + education_prop
                            + married_prop,
                            data = train.df,
                            ntree = 1000,
                            weights = as.integer(train.df$weight),
                            mtry = 3, # sprt(7) = 2.646
                            importance = TRUE)
base_forest

# Tuning
tempforest <- randomForest(run_out ~ female_prop + hispanic_prop + black_prop
                           + kids_count + elderly_count + education_prop
                           + married_prop,
                           data = train.df,
                           ntree = 1000,
                           mtry=6)

mtry <- c(1:7) # 7 variables used as X's

# Make room for m, OOB error
keeps2 <- data.frame(m = rep(NA, length(mtry)),
                     OOB_err_rate = rep(NA, length(mtry)))

for (idx in 1:length(mtry)){
  tempforest <- randomForest(run_out ~ female_prop + hispanic_prop + black_prop
                             + kids_count + elderly_count + education_prop
                             + married_prop,
                             data = train.df,
                             ntree = 1000,
                             weights = as.integer(train.df$weight),
                             mtry = mtry[idx])
  
  keeps2[idx, 'm'] <- mtry[idx]
  
  keeps2[idx, 'OOB_err_rate'] <- mean(predict(tempforest)!= train.df$run_out)
  
}
keeps2

ggplot(data = keeps2) +
  geom_line(aes(x=m, y=OOB_err_rate)) +
  theme_bw() + labs(x = 'm (mtry) value', y = 'OOB error rate')

# Best OOB had mtry = 6

final_forest <- randomForest(run_out ~ female_prop + hispanic_prop + black_prop
                             + kids_count + elderly_count + education_prop
                             + married_prop,
                             data=train.df,
                             ntree=1000,
                             weights=as.integer(train.df$weight),
                             mtry=6) # Based on tuning

### ROC Curve ###

pi_hat <- predict(final_forest, test.df, type = "prob")[,1] # POSITIVE EVENT = 1!
rocCurve <- roc(response = test.df$run_out, # Give it truth
                predictor = pi_hat, # Probabilities of positive event 1
                levels = c(0,1)) 

plot(rocCurve, print.thres = TRUE, print.auc = TRUE)

# Pi * = 0.738
# AUC = 0.755
# Specificity = 0.806 (first number in graph)
# Sensitivity = 0.654


##### General CLUSTERING to learn about data #####
library(reshape2) #For melting data frame

#Only include demographic-related columns since these will be our Xs
#Not including County since it is categorical
#Drop all NA values since we cannot have them in the data for clustering
run_out_X = drop_na(subset(run_out_data, select = c(female_prop, hispanic_prop,
                                                    black_prop, kids_count,
                                                    elderly_count, education_prop,
                                                    married_prop, hhsize)))

#Standardize the columns
run_out_stand = apply(run_out_X, 2, function(x){(x - mean(x))/sd(x)})

#Compute observation-observation distances
run_out_dist = dist(run_out_stand, method = "euclidean")

#Measure cluster-to-cluster similarity
run_out_clust = hclust(run_out_dist, method = "ward.D2")
#plot(run_out_clust, labels = run_out_data$)

#Making sense of the clusters and saving them
run_out_X$h_cluster = as.factor(cutree(run_out_clust, k=5))
run_out_X_long = melt(run_out_X, id.vars = c("h_cluster"))
head(run_out_X_long)
ggplot(data = run_out_X_long) +
  geom_boxplot(aes(x = h_cluster, y = value, fill = h_cluster)) +
  facet_wrap(~variable, scales = "free") +
  scale_fill_brewer("Cluster \nMembership", palette = "Dark2") +
  ggtitle("Hierarchical Clusters")

#Cluster 1: Non-Black/Hispanic
#Cluster 2: Hispanic
#Cluster 3: Black
#Cluster 4: Elderly Couples
#Cluster 5: Large Households


##### Lasso and Ridge Regression #####

#Make a training and testing DATA FRAME
RNGkind(sample.kind = "default")
set.seed(23591)
train.idx = sample(x = 1:nrow(run_out_data), size = .7*nrow(run_out_data))
train.df = run_out_data[train.idx,]
test.df = run_out_data[-train.idx,]

#Make a training and testing DATA MATRIX
x.train = model.matrix(run_out ~ hhsize + female_prop + hispanic_prop + black_prop
                       + kids_count + elderly_count + education_prop + 
                         married_prop, data = train.df)[,-1] 
x.test = model.matrix(run_out ~ hhsize + female_prop + hispanic_prop + black_prop
                      + kids_count + elderly_count + education_prop + 
                        married_prop, data = test.df)[,-1]

#Make VECTORS of run_out (our Y variable)
y.train = as.vector(train.df$run_out)
y.test = as.vector(test.df$run_out)

#Use cross validation to fit (LOTS OF) lasso and ridge regressions
lasso_cv_ro = cv.glmnet(x.train,
                        y.train,
                        family = binomial(link = "logit"),
                        alpha = 1,
                        weights = as.integer(train.df$weight))

ridge_cv_ro = cv.glmnet(x.train,
                        y.train,
                        family = binomial(link = "logit"),
                        alpha = 0,
                        weights = as.integer(train.df$weight))

#Choose the lambda value that minimizes out of sample error
best_lasso_lambda = lasso_cv_ro$lambda.min
best_ridge_lambda = ridge_cv_ro$lambda.min

#Final Lasso and Ridge models
final_lasso_ro = glmnet(x.train, y.train, 
                        family = binomial(link = "logit"),
                        alpha = 1,
                        lambda = best_lasso_lambda,
                        weights = as.integer(train.df$weight))
final_ridge_ro = glmnet(x.train, y.train, 
                        family = binomial(link = "logit"),
                        alpha = 0,
                        lambda = best_ridge_lambda,
                        weights = as.integer(train.df$weight))

##### Testing LASSO, RIDGE Model Performance #####

#QUANTIFY PREDICTION PERFORMANCE OF BOTH MODELS
test.df.preds = test.df %>% 
  mutate(
    lasso_pred_ro = predict(final_lasso_ro, x.test, type = "response")[,1],
    ridge_pred_ro = predict(final_ridge_ro, x.test, type = "response")[,1]
  )

#FIT ROC CURVES

lasso_rocCurve_ro = roc(response = as.factor(test.df.preds$run_out), 
                        predictor = test.df.preds$lasso_pred_ro, 
                        levels = c("0", "1")) 

ridge_rocCurve_ro = roc(response = as.factor(test.df.preds$run_out), 
                        predictor = test.df.preds$ridge_pred_ro, 
                        levels = c("0", "1")) 

plot(lasso_rocCurve_ro, print.thres = TRUE, print.auc = TRUE) # AUC = 0.654
plot(ridge_rocCurve_ro, print.thres = TRUE, print.auc = TRUE) # AUC = 0.656



#### USING CLUSTERING TO PREDICT ####

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

# find mean of run_out in each cluster
means <- cps_data %>%
  group_by(cluster) %>%
  summarize(Mean_run_out = mean(run_out, na.rm = TRUE))

means
# according to this, cluster 2 has the highest concentration of people who
# feel as though they may run out of food without being able to afford more.
# in order from most to least food insecurity (in CPS data), we have
# Cluster 2, 3, 4, 1, 5


#### CREATE LASSO and RIDGE MODELS FOR CLUSTERS ####

# remove NA's for the Y variable
run_out_data <- cps_data %>% drop_na(run_out)

# create train and test datasets

RNGkind(sample.kind = "default")
set.seed(23591)

train.idx.ro <- sample( x= c(1:nrow(run_out_data)), size=.8*nrow(run_out_data))

# make training dataframe
train.df.ro <- run_out_data[train.idx.ro,]


# make mutually exclusive testing dataframe
test.df.ro <- run_out_data[-train.idx.ro,]

#Make a training and testing DATA MATRIX
x.train.cluster = model.matrix(run_out ~ cluster, data = train.df.ro)[,-1] 
x.test.cluster = model.matrix(run_out ~ cluster, data = test.df.ro)[,-1]

#Make VECTORS of run_out (our Y variable)
y.train.cluster = as.vector(train.df.ro$run_out)
y.test.cluster = as.vector(test.df.ro$run_out)

#Use cross validation to fit (LOTS OF) lasso and ridge regressions
lasso_cluster_ro = cv.glmnet(x.train.cluster,
                             y.train.cluster,
                             family = binomial(link = "logit"),
                             alpha = 1,
                             weights = as.integer(train.df.ro$weight))

ridge_cluster_ro = cv.glmnet(x.train.cluster,
                             y.train.cluster,
                             family = binomial(link = "logit"),
                             alpha = 0,
                             weights = as.integer(train.df.ro$weight))

#Choose the lambda value that minimizes out of sample error
cluster_lasso_lambda = lasso_cluster_ro$lambda.min
cluster_ridge_lambda = ridge_cluster_ro$lambda.min

#Final Lasso and Ridge models
final_clust_lasso_ro = glmnet(x.train.cluster, y.train.cluster, 
                              family = binomial(link = "logit"),
                              alpha = 1,
                              lambda = cluster_lasso_lambda,
                              weights = as.integer(train.df.ro$weight))


final_clust_ridge_ro = glmnet(x.train.cluster, y.train.cluster, 
                              family = binomial(link = "logit"),
                              alpha = 0,
                              lambda = cluster_ridge_lambda,
                              weights = as.integer(train.df.ro$weight))

##### Testing cluster Model Performance #####

#QUANTIFY PREDICTION PERFORMANCE OF BOTH MODELS
test.df.preds = test.df.ro %>% 
  mutate(
    lasso_clust_pred_ro = predict(final_clust_lasso_ro, x.test.cluster, type = "response")[,1],
    ridge_clust_pred_ro = predict(final_clust_ridge_ro, x.test.cluster, type = "response")[,1]
  )

#FIT ROC CURVES

lasso_clust_roc_ro = roc(response = as.factor(test.df.preds$run_out), 
                         predictor = test.df.preds$lasso_clust_pred_ro, 
                         levels = c("0", "1")) 

ridge_clust_roc_ro = roc(response = as.factor(test.df.preds$run_out), 
                         predictor = test.df.preds$ridge_clust_pred_ro, 
                         levels = c("0", "1")) 


plot(lasso_clust_roc_ro, print.thres = TRUE, print.auc = TRUE)
plot(ridge_clust_roc_ro, print.thres = TRUE, print.auc = TRUE)
# for the lasso, AUC is 0.615
# for the ridge, AUC is 0.616
# These models are even worse than the trees and the forest. 







#### MAKE LASSO AND RIDGE MODELS USING BOTH CLUSTERS AND X VARS ####

# I'm also including two interaction variables: married*education
# because in synergy, these represent somebody who is stereotypically well off.
# and elderly*kids, because these represent people who are likely unable to work
# but are still mouths to feed. In synergy, they show that it is a "mixed family"
# with multiple generations.

#Make a training and testing DATA MATRIX
x.train.all = model.matrix(run_out ~ cluster + hhsize + female_prop + hispanic_prop + black_prop
                           + kids_count + elderly_count + education_prop + 
                             married_prop + married_prop*education_prop +
                             elderly_count*kids_count
                           , data = train.df.ro)[,-1] 
x.test.all = model.matrix(run_out ~ cluster+ hhsize + female_prop + hispanic_prop + black_prop
                          + kids_count + elderly_count + education_prop + 
                            married_prop + married_prop*education_prop +
                            elderly_count*kids_count , data = test.df.ro)[,-1]

#Make VECTORS of run_out (our Y variable)
y.train.all = as.vector(train.df.ro$run_out)
y.test.all = as.vector(test.df.ro$run_out)

#Use cross validation to fit (LOTS OF) lasso and ridge regressions
lasso_all_ro = cv.glmnet(x.train.all,
                         y.train.all,
                         family = binomial(link = "logit"),
                         alpha = 1,
                         weights = as.integer(train.df.ro$weight))

ridge_all_ro = cv.glmnet(x.train.all,
                         y.train.all,
                         family = binomial(link = "logit"),
                         alpha = 0,
                         weights = as.integer(train.df.ro$weight))

#Choose the lambda value that minimizes out of sample error
all_lasso_lambda = lasso_all_ro$lambda.min
all_ridge_lambda = ridge_all_ro$lambda.min

#Final Lasso and Ridge models
final_all_lasso_ro = glmnet(x.train.all, y.train.all, 
                            family = binomial(link = "logit"),
                            alpha = 1,
                            lambda = all_lasso_lambda,
                            weights = as.integer(train.df.ro$weight))


final_all_ridge_ro = glmnet(x.train.all, y.train.all, 
                            family = binomial(link = "logit"),
                            alpha = 0,
                            lambda = all_ridge_lambda,
                            weights = as.integer(train.df.ro$weight))


#QUANTIFY PREDICTION PERFORMANCE OF BOTH MODELS
test.df.preds = test.df.ro %>% 
  mutate(
    lasso_all_pred_ro = predict(final_all_lasso_ro, x.test.all, type = "response")[,1],
    ridge_all_pred_ro = predict(final_all_ridge_ro, x.test.all, type = "response")[,1]
  )


#FIT ROC CURVES

lasso_all_rocCurve_ro = roc(response = as.factor(test.df.preds$run_out), 
                            predictor = test.df.preds$lasso_all_pred_ro, 
                            levels = c("0", "1")) 

ridge_all_rocCurve_ro = roc(response = as.factor(test.df.preds$run_out), 
                            predictor = test.df.preds$ridge_all_pred_ro, 
                            levels = c("0", "1")) 

plot(lasso_all_rocCurve_ro, print.thres = TRUE, print.auc = TRUE) # AUC = 0.660
plot(ridge_all_rocCurve_ro, print.thres = TRUE, print.auc = TRUE) # AUC = 0.659


# So, to recap: the lasso model with the clusters has an AUC of 0.660.
# The ridge model with no clusters has an AUC of 0.656.
# The extra computing power of the clusters, and having to explain them,
# honestly is not worth it to get that extra 0.004 in the AUC. 

# so , the final model that we will use is the ridge model with no clusters. 

plot(ridge_rocCurve_ro, print.thres = TRUE, print.auc = TRUE) # AUC = 0.656

final_ro_model <- final_ridge_ro
#Find model coefficients
coef(final_ro_model)

#### PREDICT ####

summary(acs_data)

acs_matrix <- as.matrix(subset(acs_data, select = -c(serialno, PUMA, weight)))

acs_data$prediction <- predict(final_ro_model,acs_matrix,type="response")[,1]

summary(acs_data)


##### Aggregate ACS Data by PUMA #####
acs_data_sub <- acs_data %>% filter(elderly_count >=1)
#agg_ACS = aggregate(acs_data_sub$prediction,
#                    by = list(acs_data_sub$PUMA),
#                    FUN = weighted.mean(w = acs_data_sub$weight))
#the above may not work to take a weighted average

acs_data_sub_agg <- acs_data_sub %>% 
  group_by(PUMA) %>% 
  summarise(mean_run_out = weighted.mean(x = prediction, w = weight),
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

merged_ACS = merged_ACS %>% 
  mutate(insecure_seniors = mean_run_out*senior_population)

### Scatter Plot ###

# Scatterplot for run_out and education
ggplot(merged_ACS, aes(x=mean_education, y=mean_run_out)) +
  geom_point(size=3) +
  labs(
    title="Effect of Education Level on Ability to Afford Food",
    x = "Average Household Education Level",
    y = "Proportion of Households Unable to Afford Food"
  ) +
  geom_smooth(method="lm", se=FALSE, color = "red") +
  theme_bw()

# PUMA 1901503 has the highest percent of citizens who worry about running out
# of food, which is Des Moines City

# Trend Line Coefficient
coefficients(lm(mean_run_out ~ mean_education, data = merged_ACS))

# -0.07121883

### Table of Insecure Seniors ###

sorted_data <- merged_ACS %>% arrange(desc(insecure_seniors))
names(sorted_data)

library(knitr)
kable(sorted_data[, c("PUMA","insecure_seniors","senior_population", "mean_education")] %>% slice(1:5))

