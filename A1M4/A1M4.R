
# Assignment 1, Module 4

#########################################################
#	Libraries
#########################################################

library(survey)
library(dplyr)
library(devtools)
library(foreign)
library(tidyverse)
library(psych)
library(ggplot2)


#########################################################
#	Load data
#########################################################

# Load or import or download data here, e.g.

url <- "https://meps.ahrq.gov/mepsweb/data_files/pufs/h209dat.zip" 
           download.file(url, temp <- tempfile())  
           meps_path <- unzip(temp, exdir = tempdir())                               
   source("https://meps.ahrq.gov/mepsweb/data_stats/download_data/pufs/h209/h209ru.txt")
                                                       
unlink(temp)  # Unlink to delete temporary file                                                                     
head(h209) # view data

#########################################################
#	EDA -- What follows is an example only
#########################################################

# Place you EDA here.

var_types <- setNames(var_types, var_names)

h209 <- read_fwf(meps_path,                                 
		col_positions =                       
         fwf_positions(                    
             start = pos_start,            
               end = pos_end,          
         col_names = var_names),   
         col_types = var_types) 

save(h209, file ="h209.Rdata")
as.factor(h209$ADFLST42)
as.factor(h209$ADAGE42)
as.factor(h209$RACEV1X)
as.factor(h209$ADSEX42)
as.factor(h209$EMPST31)
as.factor(h209$EDUCYR)


h209a <- h209[!(h209$ADFLST42 == "-15" | h209$ADFLST42 == "-1"),]
h209a <- h209[!(h209$EMPST31 == "-15" | h209$EMPST31 == "-1" | h209$EMPST31 == "-8" | h209$EMPST31 == "-7"),]
h209a <- h209[!(h209$ADAGE42 == "-15" | h209$ADAGE42 == "-1"),]
h209a <- h209[!(h209a$ADSEX42 == "-15" | h209$ADSEX42 == "-1"),]
h209a <- h209[!(h209$EDUCYR == "-15" | h209$EDUCYR == "-1" | h209$EDUCYR == "-8" | h209$EDUCYR == "-7"),]
h209b <- subset(h209a, select = c(ADFLST42, ADAGE42, RACEV1X, ADSEX42, EMPST31, EDUCYR))

h209b <- h209b[!(h209b$ADFLST42 == "-15" | h209b$ADFLST42 == "-1"),]
h209b <- h209b[!(h209b$EMPST31 == "-15" | h209b$EMPST31 == "-1" | h209b$EMPST31 == "-8" | h209b$EMPST31 == "-7"),]
h209b <- h209b[!(h209b$ADAGE42 == "-15" | h209b$ADAGE42 == "-1"),]
h209b <- h209b[!(h209b$ADSEX42 == "-15" | h209b$ADSEX42 == "-1"),]
h209b <- h209b[!(h209b$EDUCYR == "-15" | h209b$EDUCYR == "-1" | h209b$EDUCYR == "-8" | h209b$EDUCYR == "-7"),]


set.seed(202209)
sample <- sample(c(TRUE,FALSE), nrow(h209c), replace = TRUE, prob = c(0.7,0.3))
train <- h209b[sample, ]
test <- h209b[!sample, ]

h209c <- h209b %>%
  mutate(flushot = case_when(ADFLST42 == 1 ~ 1,
                        ADFLST42 == 2 ~ 0,
                        TRUE ~ ADFLST42),)


#########################################################
#	LDA, QDA, KNN models
#########################################################

#########################################################
# Linear Discriminant Analysis

library(MASS)

m <- lda(ADFLST42~ ., data= train)
mlda <- m

plot(m)

lda.pred <- predict(m, test)
head(lda.pred$class)
head(lda.pred$posterior)
head(lda.pred$x)
names(lda.pred)
mean(lda.pred$class == test$ADFLST42)


#########################################################
# Quadratic Discriminant Analysis

m <- qda(ADFLST42~ ., data= train)
mqda <- m

predicted <- predict(model, test)
names(predicted)
head(predicted$class)
head(predicted$posterior)

# model accuracy
mean(predicted$class==test$ADFLST42)


#########################################################
# K-Nearest Neighbors

library(class)
set.seed(202209)
m <- knn(train, test, train$ADFLST42, k =1)
knn.pred1 <- m
table(m, test$ADFLST42)
mean(m == test$ADFLST42)

m <- knn(train, test, train$ADFLST42, k =3)
knn.pred2 <- m
table(m, test$ADFLST42)
mean(m == test$ADFLST42)

# install.packages("ROCR")
library(ROCR)
pred <- prediction(lda.pred$posterior[,1], test$ADFLST42)
perf <- performance(pred, "tpr", "fpr")
plot(perf,colorize = TRUE)

pred <- prediction(predicted$posterior[,1], test$ADFLST42)
perf <- performance(pred, "tpr", "fpr")
plot(perf,colorize = TRUE)


#########################################################
