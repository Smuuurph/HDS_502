
# Assignment 1, Module 6

# Principle Components Analysis (PCA) and K-means analysis

# The following is an example


#########################################################
#	Load data
#########################################################

# Read the data 
# https://meps.ahrq.gov/data_stats/download_data/pufs/h209/h209ru.txt
# Read in data from FYC file

library('MEPS')
library('dplyr')

data <- read_MEPS(year = 2018, type = "FYC") # 2018 FYC
data <- data %>%  dplyr::select(DUPERSID,DIABDX_M18,OBVPTR18, AGELAST, RACETHX, POVCAT18, TTLP18X, PERWT18F, INSCOV18, SEX)


# Check missing/NA data

NA.sum <- function(col){sum(is.na(col))}

NA.sum.data <- data %>% 
    summarise_all(NAsum) %>%
    tidyr::gather(key='feature', value='SumNA') %>%
    arrange(-NA.sum) %>%
    mutate(PctNa = NA.sum/nrow(data))

NA.sum.data
NA.sum2.data <- NA.sum.data %>% 
  filter(! (feature %in% c('DUPERSID', 'DIABDX_M18'))) %>%
  filter(PctNa < .45)

NA.sum2.data$feature
data.miss <- data %>% 
	dplyr::select(DUPERSID, DIABDX_M18, NA.sum2.data$feature) %>%
	filter(!is.na(DIABDX_M18))

# install.package('Amelia')
library('Amelia')
Amelia::missmap(data.miss)


# Create Variables

data <- data %>%
  mutate(DIABDX_M18 = as.numeric(DIABDX_M18)) %>%
  mutate(DIABETES = case_when(DIABDX_M18== 1 ~ 1,
                              DIABDX_M18== 2 ~ 0,
                              TRUE ~ DIABDX_M18)) %>%
  filter(DIABDX_M18 >0) %>%
  mutate(SEX = case_when(SEX == 1~ "Male",
                         SEX ==2 ~ "Female")) %>%
  mutate(RACE = case_when(RACETHX ==1 ~ "Hispanic",
                          RACETHX ==2 ~ "White",
                          RACETHX ==3 ~ "Black",
                          RACETHX ==4 ~ "Asian",
                          RACETHX ==5 ~ "Other")) %>%
  mutate(INSURANCE = case_when (INSCOV18 == 1 ~ "Private",
                                INSCOV18 == 2 ~ "Public",
                                INSCOV18 == 3 ~ "Uninsured")) %>%

  dplyr::rename(POVERTYLINE = POVCAT18) %>%
  dplyr::rename(WEIGHT=PERWT18F) %>%
  dplyr::rename(OFFICE_VISIT_PMT = OBVPTR18) %>%
  dplyr::rename(INCOME = TTLP18X) %>%
  mutate (DIABETES_FACTOR = if_else(DIABETES == 1, "1", "0")) %>%
  mutate (DIABETES_FACTOR = as.factor (DIABETES)) 

    data_diabetes <- select_all(data) %>% filter(DIABETES ==1) 
data_non_diabetes <-select_all(data) %>% filter(DIABETES <1)


#########################################################
#	EDA -- What follows is an example only
#########################################################

# Count_Query Function to review categorical data by looking at frequency counts

my_model <- data

Count_Query <- function(my_feature){
	enquo_feature <- enquo(my_feature)  
	tmp <- my_model %>%
	dplyr::select(DIABETES, !!enquo_feature) %>%
	collect() %>%
	group_by(DIABETES, !!enquo_feature) %>%
	tally () %>%
	ungroup ()
	return (tmp)
	}

Count_Query(RACE)
Count_Query(SEX)
Count_Query(INSURANCE)

my_model <- my_model %>% dplyr:: select (SEX, AGELAST, DIABETES_FACTOR, INSURANCE,RACE,DUPERSID, DIABETES,POVERTYLINE, WEIGHT,INCOME, OFFICE_VISIT_PMT) %>% 
  mutate(DIABETES_CHAR = case_when(DIABETES == 1 ~ "Diabetes",
                                   DIABETES == 0 ~ "No_Diabetes",
                                   TRUE ~ "Other"))

# Tableby function
# install.packages('arsenal')
library ('arsenal')
library ('knitr')

tableby (DIABETES ~ RACE + SEX+ INSURANCE + AGELAST+ POVERTYLINE+ WEIGHT+ OFFICE_VISIT_PMT + INCOME, data = my_model %>%
	mutate (diabetes_char=case_when(DIABETES ==1 ~ "Diabetes",
                                    DIABETES ==0 ~ "No Diabetes"))) %>%
	summary(pfootnote=TRUE) %>%
	knitr::kable()

library(GGally)
  ggally_input <- data %>% dplyr::select(AGELAST, POVERTYLINE, WEIGHT, OFFICE_VISIT_PMT, INCOME ) 
  #cor(data_summary)
  ggpairs(ggally_input,upper = list(continous = "density"), lower = list(continuous='smooth'))


#########################################################
#	PCA
#########################################################

PCA.data <- data %>% dplyr::select(AGELAST, POVERTYLINE, WEIGHT, OFFICE_VISIT_PMT, INCOME ) 
PCA.model <- prcomp(PCA.data, 
                    center=TRUE,
                    scale=TRUE)
summary(PCA.model)

PCA.diabetes <- data.diabetes %>% dplyr::select(AGELAST, POVERTYLINE, WEIGHT, OFFICE_VISIT_PMT, INCOME ) 
PCA.diabetes.model <- prcomp(PCA_diabetes, 
                    center=TRUE,
                    scale=TRUE)
summary(PCA.diabetes.model)

PCA.no.diabetes <- data.no.diabetes %>% dplyr::select(AGELAST, POVERTYLINE, WEIGHT, OFFICE_VISIT_PMT, INCOME ) 
PCA.no.diabetes.model <- prcomp(PCA.no.diabetes, 
                    center=TRUE,
                    scale=TRUE)
summary(PCA.no.diabetes.model)


#########################################################
# PCA - Plot Principal Components

# install.packages("remotes")
# remotes::install_github("vqv/ggbiplot",force=T)
 library ('ggplot2')
 library('plyr')
# install.packages('scales')
# library('scales')
 library('grid')
library('ggbiplot')


ggbiplot(PCA.model, ellipse=TRUE,labels= rownames(data))
ggbiplot(PCA.diabetes.model, ellipse=TRUE, labels= rownames(data_diabetes))
ggbiplot(PCA.no.diabetes.model, ellipse=TRUE, labels= rownames(data_non_diabetes))


# install.packages('factoextra')
library(factoextra)
fviz_pca_var(PCA.model, col.var = "cos2",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
                               repel =TRUE # Avoid text overlapping,
                              +theme_minimal()
                              )
 
fviz_pca_var(PCA.diabetes.model, col.var = "cos2",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
                               repel =TRUE # Avoid text overlapping
                               )

fviz_pca_var(PCA.no.diabetes.model, col.var = "cos2",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
                               repel =TRUE # Avoid text overlapping
                               )

#########################################################
# PCA - To determine the number of principal components to use

# Pull SD
sd.PCA <- PCA.model$sdev

# compute variance
var.PCA <- sd.PCA^2

# Proportion of variance explained
prop.PCA <- var.PCA/sum(var.PCA)

# Plot the cumulative proportion of variance explained
plot(cumsum(prop.PCA),xlab= "Principal components", ylab="Proportion of variance explained", type='b',  main="CY2018 MEPs Database")


# Diabetic Population Only
sd.PCA.diabetes <- PCA.diabetes.model$sdev
var.PCA.diabetes <- sd.PCA.diabetes^2
prop.PCA.diabetes <- var.PCA.diabetes/sum(var.PCA.diabetes)
plot(cumsum(prop.PCA.diabetes),xlab= "Principal components", ylab="Proportion of variance explained", type='b', main="CY2018 MEPs Database - Diabetic Population Only")


# Non_Diabetic Population Only
sd.PCA.no.diabetes <- PCA.no.diabetes.model$sdev
var.PCA.no.diabetes <- sd.PCA.no.diabetes^2
prop.PCA.no.diabetes <- var.PCA.no.diabetes/sum(var.PCA.no.diabetes)
plot(cumsum(prop.PCA.no.diabetes),xlab= "Principal components", ylab="Proportion of variance explained", type='b', main="CY2018 MEPs Database - Non_Diabetic Population Only")


plot(PCA.model, type="lines")
plot(PCA.diabetes.model, type="lines")
plot(PCA.no.diabetes_model, type="lines")


#########################################################
#	K-means Clustering
#########################################################

# Scale the data
data.km <- data %>% dplyr::select(AGELAST, POVERTYLINE, WEIGHT, OFFICE_VISIT_PMT, INCOME ) 
data.scaled <- scale(data.km)
head(data.scaled, n=3)


set.seed(202209)
km <- kmeans(data.scaled, 3, nstart=20)
# print(km)

#Add the cluster number back to the dataset
data.cluster <- cbind(data.scaled, km$cluster)
head(data.cluster)

fviz_cluster(km, data=data_cluster)


#########################################################
