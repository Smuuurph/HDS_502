
# Assignment 1, Module 2


#########################################################
#     Initialization
#########################################################

# Install libraries and set pathnames as desired and needed. E.g.,

# install.packages("tidyverse")
# citation("tidyverse")
library(tidyverse)

# install.packages("tidymodels")
# citation("tidymodels")
library(tidymodels)

# etc.


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


#########################################################
#	EDA
#########################################################

# Place you EDA here.


#########################################################
#	Logit models
#########################################################

# Construct your logit models here. E.g.,

m <- fitme(ejecta ~ Diameter + Big.Bin + Diameter:Big.Bin + Matern(1 | x + y), data=X, family=binomial)
m2 <- m
mdl2 <- "Layerm2"
(ms <- summary(m))
ms2 <- ms
AIC(m2)

# Another example

## LOGISTIC REGRESSION MODEL FOR HYPERTEN ##
LRModel = glm(HYPERTEN~DIABETES+as.factor(SEX)+AGE+CIGPDAY+TOTCHOL+as.factor(SEX)*DIABETES+as.factor(SEX)*CIGPDAY,family=binomial,data=FData_Baseline_NOHYP)

summary(LRModel)

## HOSMER-LEMESHOW TEST ##

hoslem.test(LRModel$y,LRModel$fitted.values)


## STANDARDIZED DEVIANCE RESIDUALS ##

fivenum(summary(LRModel)$deviance.resid)


## SCATTER PLOT OF DEVIANCE RESIDUALS VERSUS AGE ##

qplot(AGE,summary(LRModel)$deviance.resid)


## SCATTER PLOT OF DEVIANCE RESIDUALS VERSUS FITTED VALUES ##

ggplot(LRModel, aes(x=LRModel$fitted.values,y=summary(LRModel)$deviance.resid)) +
	geom_point(col='grey45') + 
	geom_smooth(col='grey45') +
	ggtitle("Scatter Plot of Residuals Versus Predicted Values") + 
	xlab("Predicted Values") + 
	ylab("Deviance Residuals") +
	theme(axis.text=element_text(size=16), axis.title=element_text(size=20), plot.title=element_text(size=24), panel.background = element_rect(fill = "grey92"))


## ODDS RATIOS AND CONFIDENCE INTERVALS ##

(exp(LRModel$coefficients))

(OddsRatios = exp(cbind(OddsRatios=coef(LRModel),confint(LRModel))))


## PREDICTION: PREDICTED VALUES USING THE CURRENT SAMPLE ##

summary(LRModel$fitted.values)


## PREDICTION: PREDICTED VALUES USING NEW DATA ##

newPredictors = as.data.frame(expand.grid(AGE=seq(30,70,5),DIABETES=c(0,1),SEX=c(1,2),CIGPDAY=c(mean(CIGPDAY,na.rm=TRUE)),TOTCHOL=c(mean(TOTCHOL,na.rm=TRUE))))

predicted_values = as.data.frame(cbind(newPredictors,predicted=predict.glm(LRModel,newdata=newPredictors,type="response")))

newPredictors = as.data.frame(expand.grid(AGE=seq(40,70,5),DIABETES=c(0,1),SEX=c(1,2),CIGPDAY=c(mean(CIGPDAY,na.rm=TRUE)),TOTCHOL=c(mean(TOTCHOL,na.rm=TRUE))))
predicted_values = as.data.frame(cbind(newPredictors,predicted=predict.glm(LRModel,newdata=newPredictors,type="response")))



## STANDARD ERRORS FOR PREDICTED VALUES ##

predictions = predict(LRModel,newdata=newPredictors,type="response",se=TRUE)
newPredictors$pred.full = predictions$fit

newPredictors$ymin = newPredictors$pred.full - 2*predictions$se.fit
newPredictors$ymax = newPredictors$pred.full + 2*predictions$se.fit

#########################################################
#	Model diagnostics
#########################################################

# The following is an example.

# install.packages("DHARMa")
# citation("DHARMa")
require("DHARMa")

sims <- simulateResiduals(m)
(ptype <- "ResVqq")
(loc <- paste0("Plots/",ver,"/",part,"/",mdl,ptype,".png") )
png(loc)
plot(sims)    # lines should match
	dev.off()
(ptype <- "ResVdia")
(loc <- paste0("Plots/",ver,"/",part,"/",mdl,ptype,".png") )
png(loc)
plotResiduals(sims, X$Diameter, quntreg=T)
	grid(col="gray")
	dev.off()
(ptype <- "ResVage")
(loc <- paste0("Plots/",ver,"/",part,"/",mdl,ptype,".png") )
png(loc)
plotResiduals(sims, X$Age, quntreg=T)
	grid(col="gray")
	dev.off()
(ptype <- "ResVnnd")
(loc <- paste0("Plots/",ver,"/",part,"/",mdl,ptype,".png") )
png(loc)
plotResiduals(sims, X$nnd, quntreg=T)
	grid(col="gray")
	dev.off()
(ptype <- "ResVBigBin")
(loc <- paste0("Plots/",ver,"/",part,"/",mdl,ptype,".png") )
png(loc)
testCategorical(sims, catPred=X$Big.Bin) # tests residuals against a categorical predictor
	dev.off()


#########################################################
#	ROC
#########################################################

# The following is an example.

OC <- data.frame(fit=round(m$fv), truth=as.numeric(m$data$ejecta)-1)

optimal.cutpoint.Youden <- optimal.cutpoints(X = fit  ~ truth , tag.healthy = 0, 
        methods = "Youden", pop.prev = NULL,  data=OC,
        control = control.cutpoints(), ci.fit = FALSE, conf.level = 0.95, trace = FALSE)

(msOC <- summary(optimal.cutpoint.Youden))
plot(optimal.cutpoint.Youden[1])

# install.packages("caret")
# citation("caret")
library("caret")
confusionMatrix(data=factor(OC$fit), reference=factor(OC$truth))

# Another example

## ROC CURVE AND AREA ##

roc(LRModel$y~LRModel$fitted.values,plot=TRUE)




## ROC CURVE: TRUE POSITIVE VERSUS FALSE POSITIVE ##

ggplot(LRModel,aes(d=LRModel$y, m=LRModel$fitted.values)) +
	geom_roc() + 
	style_roc() +
	ggtitle("ROC Curve for Logistic Prediction") + 
	xlab("False Positive") + 
	ylab("Tue Positive") +
	theme(axis.text=element_text(size=16), axis.title=element_text(size=20), plot.title=element_text(size=24), panel.background = element_rect(fill = "grey92"))


#########################################################
