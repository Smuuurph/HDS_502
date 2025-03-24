
# Assignment 1, Module 5

#########################################################
#	Libraries
#########################################################

# install.packages("dplyr")    # for data manipulation
# install.packages("devtools") # for loading "MEPS" package from GitHub
# install.packages("RNHANES")
# install.packages("caret")
# install.packages('yardstick')
# install.packages('skimr')      # alternative to glance + summary
# install.packages('rpart.plot') # better formatted plots than the ones in rpart

library('dplyr')
library('devtools')
library('RNHANES')
library('tidyverse')
library('skimr')         # alternative to glance + summary
library('rpart.plot')    # better formatted plots than the ones in rpart
library('caret')
library('yardstick')

devtools::install_github("e-mitchell/meps_r_pkg/MEPS") # easier file import
library('MEPS')


#########################################################
#	Load data
#########################################################

# Documentation on the RNHANES website:
# http://silentspringinstitute.github.io/RNHANES/

# This program includes a multiple linear regression example including:
# - Linear regression: to identify demographic factors associated with variables
# - Regression Decision trees, Bagged Trees, Random Forests, and Gradient Boosting
# to predict the outcome
# - Multiple Regression Tree Performance Comparison
 
 
# Read the data 
# https://cran.r-project.org/web/packages/RNHANES/vignettes/introduction.html
# https://www.cdc.gov/nchs/nhanes/index.htm?CDC_AA_refVal=https%3A%2F%2Fwww.cdc.gov%2Fnchs%2Fnhanes.htm
# https://rpubs.commpfoley73/529130#:~:text=Decision%20tree%20algorithms%20use%20the,used%20to%20predict%20new%20responses.


# install.packages('RNHANES')
library('RNHANES')

# Import and condition the response and predictor variables of your choice. Example follows.
DIQ <- nhanes_load_data("DIQ", "2013-2014", demographics = TRUE)

#Define response and predictors
DATA <- DIQ %>% 
     dplyr::mutate(DIABETES=case_when(DIQ010 == 1 ~ '1', DIQ010== 2 ~ "0", TRUE ~ '0'), 
          Gender = if_else(RIAGENDR ==2, "Female", "Male"), 
            Race = case_when (RIDRETH3 ==1 ~ "MEXICAN AMERICAN",
                                   RIDRETH3==2 ~ "OTHER HISPANIC",
                                   RIDRETH3==3~ "NON-HISPANIC WHITE",
                                   RIDRETH3==4 ~ "NON-HISPANIC BLACK",
                                   RIDRETH3 == 6 ~ "NON-HISPANIC ASIAN",
                                   RIDRETH3==7 ~ "OTHER",
                                   TRUE ~ "OTHER"),
        Education = case_when(DMDHREDU < 4 ~ "Lower Than College",
                                DMDHREDU == 4 ~ "College",
                                DMDHREDU == 5 ~ "Graduate",
                                TRUE ~ "OTHER")) %>%
     dplyr::select(SEQN, DID040,RIDAGEYR,Gender, INDFMIN2, Education, Race, DIQ300S) %>%
     dplyr::filter(DID040 %in% (58:212)) %>%
     dplyr::rename(AGE=RIDAGEYR,
          Hypertension=DIQ300S,
                Income=INDFMIN2,
         Diagnosed_Diabetes_Age=DID040
		 )


#########################################################
#	EDA -- What follows is an example only
#########################################################

DATA %>% dplyr::glimpse()

# Check missing/NA records 

DATA[DATA == '?'] <- NA
DATA[DATA == ''] <- NA

# install.packages('Amelia')
library('Amelia')
Amelia::missmap(as.data.frame(DATA))

SumNa <- function(col){sum(is.na(col))}
data.sum <- DATA %>%
  dplyr::summarise_all(SumNa) %>%
  tidyr::gather(key='feature', value='SumNa') %>%
  dplyr::arrange(-SumNa) %>%
  dplyr::mutate(PctNa =round(SumNa/nrow(DATA)*100,2))
data.sum

#remove NA records
data <- na.omit(DATA)
summary(data)


#count unique values for each variable
sapply(lapply(data, unique), length)


# multicollinearity Test
# install.packages('GGally')
library(GGally)
Multicollinearity_test <- data %>% select(Diagnosed_Diabetes_Age, AGE, Hypertension, Income)
cor(Multicollinearity_test)
ggpairs(Multicollinearity_test, lower = list(continuous='smooth'))


#########################################################
#	Create Train & Test Datasets
#########################################################

# The purpose of the analysis is to test the importance of the predictor variables on age. 

data <- data %>% select (-SEQN)
set.seed(202209)

Index <- createDataPartition(data$Diagnosed_Diabetes_Age, p=.8, list=FALSE)
train <-data[Index,]
 test <-data[-Index,]


#########################################################
#	Regression Decision Trees
#########################################################

# Include all selected variables into a full regression tree. 
RDT <- rpart(formula = Diagnosed_Diabetes_Age ~ .,
                             data = train,
                             method = "anova", 
                             xval = 10,
                             model = TRUE)

rpart.plot(RDT, yesno = TRUE)
printcp(RDT)
plotcp(RDT)


#########################################################
# Regression Decision Tree - Prune

RDTprune <- prune(RDT, cp = 0.033)
rpart.plot(RDTprune, yesno = TRUE)


#########################################################
# Regression Decision Tree - Prediction

pred.RDT <- predict(RDTprune, test, type = "vector")

plot(test$Diagnosed_Diabetes_Age, pred.RDT, 
     main = "Simple Regression: Predicted vs. Actual",
     xlab = "Actual",
     ylab = "Predicted")
abline(0, 1)

library('yardstick')
(RDT.rmse <- RMSE(pred = pred.RDT, obs = test$Diagnosed_Diabetes_Age))


#########################################################
#	Bagged Trees
#########################################################

# The general purpose of bagging (Bootstrap aggregation)is to reduce the variance (not bias). 

m.bag <- train(Diagnosed_Diabetes_Age ~ ., 
               data = train, 
               method = "treebag",  # for bagging
               tuneLength = 5,  # choose up to 5 combinations of tuning parameters
               metric = "RMSE",  # evaluate hyperparamter combinations with RMSE
               trControl = trainControl(
                 method = "cv",  # k-fold cross validation
                 number = 10,  # 10 folds
                 savePredictions = "final"       # save predictions
                 )
               )
m.bag
plot(varImp(m.bag), main="Variable Importance with Regression Bagging")

pred.bag <- predict(m.bag, test, type = "raw")
plot(test$Diagnosed_Diabetes_Age, pred.bag, 
     main = "Bagging Regression: Predicted vs. Actual",
     xlab = "Actual",
     ylab = "Predicted")
abline(0, 1)

bag.rmse <- RMSE(pred = pred.bag, obs = test$Diagnosed_Diabetes_Age)

rm(pred.bag)


#########################################################
#	Random Forests
#########################################################

# Bagged trees are a special case of random forests. 

m.forest <- train(Diagnosed_Diabetes_Age ~ ., 
               data = train, 
               method = "ranger",  # for random forest
               tuneLength = 5,  # choose up to 5 combinations of tuning parameters
               metric = "RMSE",  # evaluate hyperparamter combinations with RMSE
               trControl = trainControl(
                 method = "cv",  # k-fold cross validation
                 number = 10,  # 10 folds
                 savePredictions = "final"       # save predictions
                 )
               )
m.forest
plot(m.forest)

pred.forest <- predict(m.forest, test, type = "raw")
plot(test$Diagnosed_Diabetes_Age, pred.forest, 
     main = "Random Forest Regression: Predicted vs. Actual",
     xlab = "Actual",
     ylab = "Predicted")
abline(0, 1)

(forest.rmse <- RMSE(pred = pred.forest, obs = test$Diagnosed_Diabetes_Age))

rm(pred.forest)


#########################################################
#	Gradient Boosting Regression
#########################################################

# Boosting is a method to improve weak learners sequentially.

m.gbm <- train(Diagnosed_Diabetes_Age ~ ., 
                      data = train, 
                      method = "gbm",  # for gbm
                      tuneLength = 5,  # choose up to 5 combinations of tuning parameters
                      metric = "RMSE",  # evaluate hyperparamter combinations with ROC
                      trControl = trainControl(
                        method = "cv",  # k-fold cross validation
                        number = 10,  # 10 folds
                        savePredictions = "final",       # save predictions
                        verboseIter = FALSE,
                        returnData = FALSE
                        )
                      )
m.gbm
plot(m.gbm)

pred.gbm <- predict(m.gbm, test, type = "raw")
plot(test$Diagnosed_Diabetes_Age, pred.gbm, 
     main = "Gradient Boosing Regression: Predicted vs. Actual",
     xlab = "Actual",
     ylab = "Predicted")
abline(0,1)


(gbm.rmse <- RMSE(pred = pred.gbm, obs = test$Diagnosed_Diabetes_Age))
rm(pred.gbm)


# Summary

rbind(data.frame(model = "Manual ANOVA", RMSE = round(RDT.rmse, 5)), 
      data.frame(model = "Bagging", RMSE = round(bag.rmse, 5)),
      data.frame(model = "Random Forest", RMSE = round(forest.rmse, 5)),
      data.frame(model = "Gradient Boosting", RMSE = round(gbm.rmse, 5))
) %>% arrange(RMSE)


#########################################################
