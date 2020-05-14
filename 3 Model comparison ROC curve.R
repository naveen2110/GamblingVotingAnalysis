#############################################################################################
## Preparation

# install.packages('ROSE')

rm(list=ls()); gc(); dev.off(dev.list()["RStudioGD"])
library(rpart)
library(rpart.plot)
require(class)
library(dplyr)
library(ROSE)

setwd()
dat = read.csv('Dataset_ROC.csv', stringsAsFactors = T, head = T) # import csv is the same as after datacleaning above


##### Logistic regression ######
# Model Selection: "forward"  ##
# "backward", or "both"       ##
mod = "forward"               ##
# Ballot_Type                 ##
Ballot_Type_Selection = 1     ##
##### CART #####################
# complexity parameter        ##
CPP = 1e-3                    ##
# Minsplit                    ##
MS = 5                        ##
##### KNN ######################
# Kgrid                       ##
grid = seq(1, 50, 2)          ##
################################
# Ballot_Type                 ##
Ballot_Type_selecton = 1      ##
# partition ratio             ##
pratio = 0.6                  ##
# round final tables          ##
rounding = 3                  ##
# Cut-off point               ##
cutoff = 0.6                  ##
# seed                        ##
seed = 1                      ##
################################

# Data separation
dat.ballot = subset(dat, Ballot_Type == Ballot_Type_Selection)

# Partition index
set.seed(seed)
n.train = floor(nrow(dat.ballot) * pratio)
ind.train = sample(1:nrow(dat.ballot), n.train)
ind.test = setdiff(1:nrow(dat.ballot), ind.train)

# selected column

selected_col = c(
  'Voted_for_or_Not',
  'Population',
  'Per_Capita_Income',
  'Medium_Family_Income',
  'Size_of_County',
  'Population_Density',
  'No_of_Churches',
  'No_of_Church_Members',
  'Poverty_Level_Rate',
  'Unemployment_Rate',
  'Percent_White',
  'Percent_Black',
  'Percent_Other',
  'Percent_Minority',
  'Percent_Male',
  'Percent_Female',
  'Age_Less_than_18',
  'Age_18_24',
  'Age_24_44',
  'Age_44_64',
  'Age_Older_than_64'
)


#############################################################################################
#############################################################################################
# Logistic regression

cleaned_dat = dat.ballot[, selected_col]

# scaling
cleaned_dat[, 2:21] = scale(cleaned_dat[, 2:21])

# Partition data
dat_train = cleaned_dat[ind.train, ]
dat_test = cleaned_dat[ind.test, ]


min.model = glm(Voted_for_or_Not ~ 1, data = dat_train, family = 'binomial')
max.model = glm(Voted_for_or_Not ~ ., data = dat_train, family = 'binomial')
max.formula = formula(max.model)

obj = step(min.model, direction = mod, scope = max.formula)

get.or = function(sobj, alpha = .05) {
  b = sobj$coef[-1, 'Estimate']
  se.b = sobj$coef[-1, 'Std. Error']
  pval = sobj$coef[-1, 'Pr(>|z|)']
  or = exp(b)
  se.or = exp(b) * se.b
  lb = b - qnorm(alpha / 2) * se.b
  lb.or = exp(lb)
  ub = b + qnorm(1 - alpha / 2) * se.b
  ub.or = exp(ub)
  out = cbind(or, se.or, lb, ub, pval)
  colnames(out) = c('OR',
                    'SE',
                    paste0((1 - alpha) * 100, '% CI, lower'),
                    paste0((1 - alpha) * 100, '% CI, upper'),
                    'p value')
  return(out)
}

yhat = predict(obj, newdata = dat_test, type='response')

dichotomize = function(yhat, cutoff=cutoff) {
  out = rep(0, length(yhat))
  out[yhat > cutoff] = 1
  out
}

LOG.yhat.class = dichotomize(yhat, .1)
log_err = mean(LOG.yhat.class != dat_test$Voted_for_or_Not)


#############################################################################################
## CART

PICERGA_dat = dat.ballot[, selected_col]

# Partition data
PICERGA_train = PICERGA_dat[ind.train, ]
PICERGA_test = PICERGA_dat[ind.test, ]

##########################
## PICERGA - Min. Error Tree
PICERGA_fit_minerr = rpart(
  Voted_for_or_Not ~ .,
  method = "class",
  data = PICERGA_train,
  cp = CPP,
  minsplit = MS
)
PICERGA_bestcp = PICERGA_fit_minerr$cptable[which.min(PICERGA_fit_minerr$cptable[, "xerror"]), "CP"]
PICERGA_tree_minerr = prune(PICERGA_fit_minerr, cp = PICERGA_bestcp)
# prediction by the Min. Error Tree
PICERGA_yhat_minerr = predict(PICERGA_tree_minerr, PICERGA_test, type = "prob")[, 2]
# set compare with cut-off value
PICERGA_yhat_minerr_cutoff = as.numeric(PICERGA_yhat_minerr > cutoff)
# error of the Min. Error Tree
PICERGA_err_minerr = 1 - mean(PICERGA_yhat_minerr_cutoff == PICERGA_test$Voted_for_or_Not)

#############################################################################################
## KNN

#Best KNN function
get.prob = function(x) {
  prob = attr(x, 'prob')
  cl = as.numeric(x)
  ind = which(cl == 1)
  prob[ind] = 1 - prob[ind]
  return(prob)
}

knn.bestK = function(train, test, y.train, y.test, k.grid = grid, ct = 0.5) {
  # browser() 
  #a way to debug
  fun.tmp = function(x) {
    #browser()
    y.tmp = knn(train, test, y.train, k = x, prob=T) # run knn for each k in k.grid
    prob = get.prob(y.tmp)
    y.hat = as.numeric( prob > ct ) + 1
    return( sum(y.hat != as.numeric(y.test)) )
  }
  ## create a temporary function (fun.tmp) that we want to apply to each value in k.grid
  error = unlist(lapply(k.grid, fun.tmp))
  names(error) = paste0('k=', k.grid)
  ## it will return a list so I need to unlist it to make it to be a vector
  out = list(k.optimal = k.grid[which.min(error)],
             error.min = min(error)/length(y.test),
             error.all = error/length(y.test))
  return(out)
}

Overall_dat = dat.ballot[,selected_col]
Overall_dat[,2:ncol(Overall_dat)] = scale(Overall_dat[,2:ncol(Overall_dat)])

# partation data
Overall_Xtrain = Overall_dat[ind.train,2:ncol(Overall_dat)]
Overall_ytrain = Overall_dat[ind.train,1]
Overall_Xtest = Overall_dat[ind.test,2:ncol(Overall_dat)]
Overall_ytest = Overall_dat[ind.test,1]

# knn over Age
Overall_best_knn = knn.bestK(Overall_Xtrain, Overall_Xtest, Overall_ytrain, Overall_ytest, k.grid = grid, cutoff)

# rerun with the best k
Overall_ypred = knn(Overall_Xtrain, Overall_Xtest, Overall_ytrain, k=Overall_best_knn$k.optimal, prob=T)

KNN_err = mean(Overall_ypred != Overall_ytest)

#############################################################################################
## ROC Curve

Logistic_reg_ROC = roc.curve(dat_test$Voted_for_or_Not, LOG.yhat.class)
CART_ROC = roc.curve(PICERGA_test$Voted_for_or_Not, PICERGA_yhat_minerr_cutoff)
KNN_ROC = roc.curve(as.numeric(Overall_ytest), as.numeric(Overall_ypred))

all_ROC = matrix(1:9, nrow = 3, ncol = 3)
rownames(all_ROC) = c("Logistic regession",'CART','KNN')
colnames(all_ROC) = c('Area Under the curve (the large the better)','Error','Cutoff')
all_ROC["Logistic regession",] = c(Logistic_reg_ROC$auc,log_err,cutoff)
all_ROC["CART",] = c(CART_ROC$auc,PICERGA_err_minerr,cutoff)
all_ROC["KNN",] = c(KNN_ROC$auc,KNN_err,cutoff)

sorted_all_ROC = all_ROC[order(all_ROC[,1], decreasing = T),] %>% round(rounding)
View(sorted_all_ROC)
