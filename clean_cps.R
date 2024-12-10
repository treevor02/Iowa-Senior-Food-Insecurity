rm(list=ls())

library(tidyverse) #includes ggplot2 and dplyer
library(ggthemes) #optional but classy ;)
library(logistf)#firth's penalized
library(glmnet) # for fitting lasso, ridge regressions (GLMs)
library(haven) #for reading in SAS data exports
library(randomForest) #for the random forest

cps<- read.csv("data/cps_00005.csv")
head(cps)

table(cps$ FSWROUTY)

# begin cleaning: 

# clean x variables: 
cps <- cps %>%
  mutate(SEX = SEX- 1 , # Create dummy variables
         CHILD = ifelse(AGE < 18, 1, 0),
         ELDERLY = ifelse(AGE > 60, 1, 0), #NOTE DEFINITION
         BLACK = ifelse(RACE==200, 1, 0),
         HISPANIC = ifelse(HISPAN>0, 1, 0),
         EDUC = as.integer(EDUC %in% c(91,92,111,123,124,125)),
         EMP = as.integer(EMPSTAT %in% c(1,10,12)),
         MARRIED = as.integer(MARST %in% c(1,2)),
         DIFF = ifelse(DIFFANY==2, 1, 0),
         COUNTY = as.factor(COUNTY))

# currently, one row of cps = one individual
# however, we want to make prediction on the family level
# aggregate to the family level - this is where we choose FAMILY-LEVEL traits
# that we want to calculate. For example, household size is equal to the
# number of rows for that family.

cps_data <- cps %>%
  group_by(CPSID = as.factor(CPSID)) %>%
  summarise(COUNTY = first(COUNTY),
            #family level weight
            weight = first(HWTFINL),
            #household size
            hhsize = n(),
            #Y variables - i.e., measures of hunger
            #see CPS website for details
            #FSSTATUS, etc. is the same for each member -just take first value for each family
            FSTOTXPNC_perpers = FSTOTXPNC/hhsize, # In per person terms
            FSSTATUS = first(FSSTATUS),
            FSSTATUSMD = first(FSSTATUSMD),
            FSFOODS = first(FSFOODS),
            FSWROUTY = first(FSWROUTY),
            FSBAL = first(FSBAL),
            FSRAWSCRA = first(FSRAWSCRA),
            FSTOTXPNC = first(FSTOTXPNC),
            FSSTATUS = first(FSSTATUS),
            #count of family members in various categories
            female_prop = sum(SEX)/hhsize,
            hispanic_prop = sum(HISPANIC)/hhsize,
            black_prop= sum(BLACK)/hhsize,
            kids_count= sum(CHILD),
            elderly_count= sum(ELDERLY),
            education_prop= sum(EDUC)/hhsize,
            married_prop= sum(MARRIED)/hhsize) %>% ungroup()

# we chose to make female, hispanic, black, education, and married a proportion.
# this is because they represent the "makeup" of the family rather than the number.

# we kept kids and elderly as a count, because more people unable to work = more need. 


# now, each row of cps_data is a FAMILY
#note... we just calculated the number of people in each family that belong

summary(cps_data) # see extremes for food security variables
cps_data$married

#https://cps.ipums.org/cps-action/variables/search

# clean y variables:

cps_data <- cps_data %>%
  mutate(FSSTATUS = ifelse(FSSTATUS %in% c(98,99), NA, FSSTATUS),
         FSSTATUSMD = ifelse(FSSTATUSMD %in% c(98,99), NA, FSSTATUSMD),
         FSFOODS = ifelse(FSFOODS %in% c(98,99), NA, FSFOODS),
         FSWROUTY = ifelse(FSWROUTY %in% c(96,97,98,99), NA, FSWROUTY),
         FSBAL = ifelse(FSBAL %in% c(96,97,98,99), NA, FSBAL),
         FSRAWSCRA = ifelse(FSRAWSCRA %in% c(98,99), NA, FSRAWSCRA),#raw score
         FSTOTXPNC = ifelse(FSTOTXPNC %in% c(999), NA, FSTOTXPNC)) %>%
  mutate(FSSTATUS = ifelse(FSSTATUS > 1, 1, 0),
         FSSTATUSMD = ifelse(FSSTATUSMD > 1, 1, 0),
         FSFOODS = ifelse(FSFOODS > 1, 1, 0),
         FSWROUTY = ifelse(FSWROUTY > 1, 1, 0),#more missings
         FSBAL = ifelse(FSBAL > 1, 1, 0),
         FSRAWSCRA=ifelse(FSRAWSCRA > 1, 1, 0))

# rename the y variables that we are going to use.
# FSWROUTY is run_out because it measures the worry that food will run out before you are able to afford more
# FSBAL is not_balanced because it measures people being unable to afford to eat balanced meals in the past year
colnames(cps_data) = c("CPSID", "COUNTY", "weight", "hhsize", "FSTOTXPNC_perpers",
                       "FSSTATUS", "FSSTATUSMD", "FSFOODS", "run_out",
                       "not_balanced", "FSRAWSCRA", "FSTOTXPNC", "female_prop",
                       "hispanic_prop", "black_prop", "kids_count",
                       "elderly_count", "education_prop", "married_prop")

str(cps_data)
summary(cps_data)

#Note: many of our y variables contain some NA values.
#Do not use complete.cases or na.omit on the whole dataset.



