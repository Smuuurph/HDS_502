
# Assignment 1, Module 3

#########################################################
#     Initialization
#########################################################

# Install libraries and set pathnames as desired and needed. E.g.,

library(NHANES)
df = NHANES

#########################################################
#     Functions
#########################################################

# Place your own functions here if desired. E.g.,

WriteCSV <- function(CSVfileName, RdataSet) {
	outfile <- paste(WD, paste(CSVfileName, ".csv", sep=""), sep="/")
	write.csv(RdataSet, file=outfile, row.names=F)
	}

#########################################################
#	Load data
#########################################################

# Load or import or download data here, e.g.
# Download the data including demographics data (which includes sample weights).

dat <- nhanes_load_data("EPHPP", "2013-2014", demographics = TRUE)
str(dat)


#########################################################
#	EDA
#########################################################

# Place you EDA here.
# EDA code here. Select continuous response and choose predictor variables.



# Find the sample size for urinary triclosan. This is an example.
nhanes_sample_size(dat,
  column = "URXTRS",
  comment_column = "URDTRSLC",
  weights_column = "WTSB2YR")

# Compute the detection frequency of urinary triclosan
nhanes_detection_frequency(dat,
  column = "URXTRS",
  comment_column = "URDTRSLC",
  weights_column = "WTSB2YR")

# Compute 95th and 99th quantiles for urinary triclosan
nhanes_quantile(dat,
  column = "URXTRS",
  comment_column = "URDTRSLC",
  weights_column = "WTSB2YR",
  quantiles = c(0.95, 0.99))

library("survey")
design <- svydesign(
	id = ~SEQN,
	weights = ~WTSB2YR,
	data = dat
	)

svyquantile(~URXTRS, design, c(0.95, 0.99))

# Plot a histogram of triclosan distribution
nhanes_hist(dat,
  column = "URXTRS",
  comment_column = "URDTRSLC",
  weights_column = "WTSB2YR")



#########################################################
#	Regression models
#########################################################

# Subset to include only model variables, following is and example only, use your choices
# Be sure your first variable is your response variable. The remaining code then will work.

X <- data.frame(URXTRS=dat$URXTRS, age=dat$RIDAGEYR, income=dat$INDHHIN2, gender=dat$RIAGENDR)
X$gender <- factor(X$gender)
str(X)


#########################################################
# CHECK FOR MULTICOLLINEARITY
require("GGally")
ggpairs(X, aes(alpha = 0.4))
ggsave("")  # insert path and/or just filename
require("Hmisc")
rcorr(as.matrix(X), type="pearson")
require("car")
m <- lm(URXTRS ~ age + income + gender, data=X)
mfull <- m
vif(m)
m.anova <- anova(m)


#########################################################
# ALL SUBSETS SELECTION
# install.packages("leaps")
# citation("leaps")
require("leaps")
m <- regsubsets(URXTRS ~ (age*income*gender)^2,
               data = X,
               nbest = 1,       # 1 best model for each number of predictors
               nvmax = NULL,    # NULL for no limit on number of variables
               force.in = NULL,
               force.out = NULL,
               method = "exhaustive")
msubsets <- m
(ms <- summary(m))
which.max(ms$adjr2)      # best model
max(round(ms$adjr2,2))   # highest adjusted R^2 for best model
which.min(sqrt(ms$rss))      # best model
min(round(sqrt(ms$rss),2)) # lowest mse
# why the diference?


#########################################################
# STEPWISE SELECTION
m <- step(lm(URXTRS ~ (age*income*gender)^2, data=X))
mstep <- m
summary(m)
sum(m$residuals^2) / m$df.residual    # best mse


#########################################################
# RIDGE REGRESSION
require(MASS)
m <- lm.ridge(URXTRS ~ (age*income*gender)^2, data=X, lambda = seq(-100,-50,1))
mridge <- m
#      generalized cross-validation scores
require(broom)
tm <- tidy(m)
head(tm)
(gm <- glance(m))
require(ggplot2)
ggplot(tm, aes(lambda, estimate, color = term)) + geom_line()  # ests stabilize at ~ 100
 ggsave("Plots/RoofingRidge1.png")
ggplot(tm, aes(lambda, GCV)) + geom_line() + geom_vline(xintercept = gm$lambdaGCV, col = "red", lty = 2)
# ggsave("Plots/RoofingRidge2.png")

#    Compare ridge to non-ridge coefs. lambda ~ 0 => no multicolinearity.
#    When ridge coefs close to non-ridge coefs, ridge model is viable
#    Note: ridge regression centers the data so the coefs won't match reduced model
round(m$coef[, which(m$lambda == gm$lambdaGCV)], 2)   # ridge
round(m$coef[, which(m$lambda == 0)], 2)              # non-ridge


#########################################################
# LASSO
# install.packages("glmnet")
# citation("glmnet")
require(glmnet)
H <- X
X <- na.omit(H)
x <-X[, -1]
x <- matrix(as.numeric(unlist(x)),nrow=nrow(x))
y <- X$URXTRS
m <- cv.glmnet(x, y)
mlasso <- m
#   Plot shows a large range of lambda at minimum MSE.
#   Min lambda gives "best" model with number of effect at the top.
 png("")
plot(m);grid(col="gray")  # shows range of lambda used to find the "best" model. Small MSE is good.
 dev.off()
m$lambda.1se              # lambda to obtain "best" model
(pred <- predict(m, s = mlasso$lambda.min, type = 'coefficients')[1:4,])  # best model


#########################################################
#	Model diagnostics
#########################################################

# Generate model diagnostics on your chosen regression model.

# install.packages("DHARMa")
# citation("DHARMa")
require("DHARMa")

# Be sure your chosen model is in "m".
# E.g., m <- mstep or m <- mlasso, etc. You may need to run for DHARMa to run:
# m <- lm(your best model,data=X)
# for (i in 1:4) {m$coefficents[i] <- pred[i]}
sims <- simulateResiduals(m)

# Residuals analysis
(ptype <- "ResVqq")
(loc <- paste0(".png") )
png(loc)
plot(sims)    # lines should match
	dev.off()

# Adapt for continuous variables
(ptype <- "ResVage")
(loc <- paste0(".png") )
png(loc)
plotResiduals(sims, X$age, quntreg=T)
	grid(col="gray")
	dev.off()

# Adapt for categorical variables
(ptype <- "ResVBigBin")
(loc <- paste0(".png") )
png(loc)
testCategorical(sims, catPred=X$gender) # tests residuals against a categorical predictor
	dev.off()


#########################################################

