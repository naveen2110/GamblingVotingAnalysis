#############################################################################################
#############################################################################################
## Preparation

rm(list=ls()); gc(); dev.off(dev.list()["RStudioGD"])
library(rpart)
library(rpart.plot)
library(ROSE)

setwd()
dat = read.csv('Dataset_LOG.csv', stringsAsFactors=T, head=T) # import csv is the same as after datacleaning above

# you can adjust the input of this part (ballot = 1 or ballot = 2) 
################################
# Model Selection: "forward"  ##
# "backward", or "both"       ##
mod = "forward"               ##
# partition ratio             ##
pratio = 0.6                  ##
# Cut-off point               ##
CO = 0.6                      ##
# Ballot_Type                 ##
Ballot_Type_Selection = 1     ##
################################

# Data separation
dat.ballot = subset(dat, Ballot_Type == Ballot_Type_Selection)

# Partition index
set.seed(1)
n.train = floor(nrow(dat.ballot)*pratio)
ind.train = sample(1:nrow(dat.ballot), n.train)
ind.test = setdiff(1:nrow(dat.ballot), ind.train)
#############################################################################################
#############################################################################################

cleaned_dat = dat.ballot[,c('Voted_for_or_Not', 'Population','Per_Capita_Income',
                            'Medium_Family_Income',	'Size_of_County','Population_Density',
                            'No_of_Churches','No_of_Church_Members',
                            'Poverty_Level_Rate','Unemployment_Rate',
                            'Percent_White', 'Percent_Black', 'Percent_Other', 'Percent_Minority',
                            'Percent_Male', 'Percent_Female',
                            'Age_Less_than_18', 'Age_18_24', 'Age_24_44', 
                            'Age_44_64', 'Age_Older_than_64')]

# Partition data
dat_train = cleaned_dat[ind.train,]
dat_test = cleaned_dat[ind.test,]


min.model = glm(Voted_for_or_Not ~ 1, data = dat_train, family = 'binomial')
max.model = glm(Voted_for_or_Not ~ ., data = dat_train, family = 'binomial')
max.formula = formula(max.model)

obj = step(min.model, direction=mod, scope=max.formula) # it will print out models in each step
summary(obj) # it will give you the final model


get.or = function(sobj, alpha=.05) {
  b = sobj$coef[-1, 'Estimate']
  se.b = sobj$coef[-1, 'Std. Error']
  pval = sobj$coef[-1, 'Pr(>|z|)']
  or = exp(b); se.or = exp(b)*se.b
  lb = b - qnorm(alpha/2)*se.b; lb.or = exp(lb)
  ub = b + qnorm(1-alpha/2)*se.b; ub.or = exp(ub)
  out = cbind(or, se.or, lb, ub, pval)
  colnames(out) = c('OR', 'SE', paste0((1-alpha)*100, '% CI, lower'),
                    paste0((1-alpha)*100, '% CI, upper'), 'p value')
  return(out)
}
get.or(summary(obj))


yhat = predict(obj, newdata = dat_test, type='response')
hist(yhat)

dichotomize = function(yhat, cutoff=CO) {
  out = rep(0, length(yhat))
  out[yhat > cutoff] = 1
  out
}

yhat.class = dichotomize(yhat, .1)
err = mean(yhat.class != dat_test$Voted_for_or_Not) # misclassification error rate


table(dat_test$Voted_for_or_Not, yhat.class)

sen = function(ytrue, yhat) {
  ind.true1 = which(ytrue == 1)
  mean( ytrue[ind.true1] == yhat[ind.true1] )
}

spe = function(ytrue, yhat) {
  ind.true0 = which(ytrue == 0)
  mean( ytrue[ind.true0] == yhat[ind.true0] )
}

sen(dat_test$Voted_for_or_Not, yhat.class)
spe(dat_test$Voted_for_or_Not, yhat.class)

Logistic_reg_ROC = roc.curve(dat_test$Voted_for_or_Not, yhat.class)
print(Logistic_reg_ROC$auc)
print(err)

