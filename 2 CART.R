#############################################################################################
#############################################################################################
## CART Classification
#  this file includes analysis running CART over different factors to classify 
#  whether a group of people or country will vote for voted for or not
#  and compare which factors are the best ones to use.
# Content as follows:
# Preparation
# 0. All predictors in dataset
# 1. Population, Income, Church, Economy, Race, Gender, and Age predictors
# 2. Income predictors
# 3. Church predictors
# 4. Economy predictors
# 5. Race predictors
# 6. Age predictors
# 7. Gender predictors
# Error Comparison of predictor groups
#############################################################################################
#############################################################################################
## Preparation

# install.packages('ROSE')

rm(list=ls()); gc(); dev.off(dev.list()["RStudioGD"])
library('rpart')
library('rpart.plot')
library('dplyr')
library('ROSE')

setwd()
dat = read.csv('Dataset_CART.csv', stringsAsFactors=T, head=T) # import csv is the same as after datacleaning above

#############################################################################################
#############################################################################################
# Please run code from here after you make adjustment.
################################
# partition ratio             ##
pratio = 0.5                  ##
# complexity parameter        ##
CPP = 1e-3                    ##
# Minsplit                    ##
MS = 5                        ##
# Cut-off point               ##
cutoff = 0.6                  ##
# Ballot_Type                 ##
Ballot_Type_Selection = 1     ##
# round final tables          ##
rounding = 4                  ##
################################

# Data separation
dat.ballot = subset(dat, Ballot_Type == Ballot_Type_Selection)


#############################################################################################
#############################################################################################

## 0. All predictors in dataset

## Which states voted for/against (just for overview)?
fit.ballot = rpart(Voted_for_or_Not ~ ï..State_No, method="class", data=dat.ballot, minsplit=MS)

# Plot  Full Tree
rpart.plot(fit.ballot, main = paste('Voting Outcome (1 = In Favor) per State for Ballot', Ballot_Type_Selection))
# 
# remove State_No, County_No, Ballot_Type, Votes_For, Votes_Against, and Total_Votes for overall analysis
dat.ballot_skimmed <- dat.ballot[ -c(1:7) ]

# Partition index
set.seed(1)
n.train = floor(nrow(dat.ballot_skimmed)*0.6)
ind.train = sample(1:nrow(dat.ballot_skimmed), n.train)
ind.test = setdiff(1:nrow(dat.ballot_skimmed), ind.train)

# Partition data
fit_train = dat.ballot_skimmed[ind.train,]
fit_test = dat.ballot_skimmed[ind.test,]

## Minimum Error Tree for all Predictors
fit_minerr = rpart(Voted_for_or_Not ~ ., method="class", data=dat.ballot_skimmed, cp=CPP, minsplit=MS)
fit_bestcp = fit_minerr$cptable[which.min(fit_minerr$cptable[,"xerror"]),"CP"]
fit_tree_minerr = prune(fit_minerr, cp = fit_bestcp)
## plot the Min. Error Tree 
rpart.plot(fit_tree_minerr, main = paste('The Min. Error Tree for All Predictors in Ballot', Ballot_Type_Selection))
# prediction by the Min. Error Tree
fit_yhat_minerr = predict(fit_tree_minerr, fit_test, type = "prob")[,2]
# set compare with cut-off value
fit_yhat_minerr_cutoff = as.numeric(fit_yhat_minerr > cutoff)
# error of the Min. Error Tree
fit_err_minerr = 1- mean(fit_yhat_minerr_cutoff != fit_test$Voted_for_or_Not)
fit_err_minerr
# sensitifity and sepcificity
fit_minerr_table = table(fit_test$Voted_for_or_Not, fit_yhat_minerr_cutoff)
if (ncol(fit_minerr_table) == 1) {
  fit_minerr_spe = fit_minerr_table[1,1]/sum(fit_minerr_table[1,1],0)
  fit_minerr_sen = 0/sum(fit_minerr_table[2,1],0)
  fit_minerr_acc = 1-sum(0,fit_minerr_table[2,1])/sum(fit_minerr_table)
} else {
  fit_minerr_spe = fit_minerr_table[1,1]/sum(fit_minerr_table[1,1],fit_minerr_table[1,2])
  fit_minerr_sen = fit_minerr_table[2,2]/sum(fit_minerr_table[2,1],fit_minerr_table[2,2])
  fit_minerr_acc = 1-sum(fit_minerr_table[1,2],fit_minerr_table[2,1])/sum(fit_minerr_table)
}
fit_minerr_table

## Best Pruned Tree for All Predictors

# Find standard error, and match the standard error to its CP
fit_mincp_SD = fit_minerr$cptable[which.min(fit_minerr$cptable[,"xerror"]),"xstd"]
fit_mincp_SE = fit_mincp_SD / sqrt(nrow(fit_train))
fit_pruned_SD = fit_mincp_SD - fit_mincp_SE
fit_mincp_SD_less_split = fit_minerr$cptable[,'xstd'][1:which.min(fit_minerr$cptable[,"xerror"])]
fit_pruned_CP = fit_minerr$cptable[which(abs(fit_mincp_SD_less_split-fit_pruned_SD)==min(abs(fit_mincp_SD_less_split-fit_pruned_SD))),"CP"]

# get the best pruned tree
fit_best_pruned = prune(fit_minerr, cp = fit_pruned_CP)
# plot the best pruned Tree 
rpart.plot(fit_best_pruned, main = paste('The Best Pruned Tree for All Predictors in Ballot', Ballot_Type_Selection))
# prediction by the best pruned Tree
fit_yhat_best_pruned = predict(fit_best_pruned, fit_test, type = "prob")[,2]
# set compare with cut-off value
fit_yhat_pruned_cutoff = as.numeric(fit_yhat_best_pruned > cutoff)
# error of the best pruned tree
fit_err_best_pruned = 1 - mean(fit_yhat_pruned_cutoff == fit_test$Voted_for_or_Not)
# sensitifity and sepcificity
fit_best_pruned_table = table(fit_test$Voted_for_or_Not, fit_yhat_pruned_cutoff)
if (ncol(fit_best_pruned_table) == 1) {
  fit_best_pruned_spe = fit_best_pruned_table[1,1]/sum(fit_best_pruned_table[1,1],0)
  fit_best_pruned_sen = 0/sum(fit_best_pruned_table[2,1],0)
  fit_best_pruned_acc = 1-sum(0,fit_best_pruned_table[2,1])/sum(fit_best_pruned_table)
} else {
  fit_best_pruned_spe = fit_best_pruned_table[1,1]/sum(fit_best_pruned_table[1,1],fit_best_pruned_table[1,2])
  fit_best_pruned_sen = fit_best_pruned_table[2,2]/sum(fit_best_pruned_table[2,1],fit_best_pruned_table[2,2])
  fit_best_pruned_acc = 1-sum(fit_best_pruned_table[1,2],fit_best_pruned_table[2,1])/sum(fit_best_pruned_table)
}
fit_best_pruned_table

## Full Tree for All Predictors
fit_full <- rpart(Voted_for_or_Not ~ ., data = fit_train,method="class", cp = CPP, minsplit=MS)
## plot the full tree 
rpart.plot(fit_full, main = paste('The Full Tree for All Predictors in Ballot', Ballot_Type_Selection))
# prediction by the full tree
fit_yhat_full = predict(fit_full, fit_test, type = "prob")[,2]
# set compare with cut-off value
fit_yhat_full_cutoff = as.numeric(fit_yhat_full > cutoff)
# error of the full tree
fit_err_full = 1 - mean(fit_yhat_full_cutoff == fit_test$Voted_for_or_Not)
# sensitifity and sepcificity
fit_full_table = table(fit_test$Voted_for_or_Not, fit_yhat_full_cutoff)
if (ncol(fit_full_table) == 1) {
  fit_full_spe = fit_full_table[1,1]/sum(fit_full_table[1,1],0)
  fit_full_sen = 0/sum(fit_full_table[2,1],0)
  fit_full_acc = 1-sum(0,fit_full_table[2,1])/sum(fit_full_table)
} else {
  fit_full_spe = fit_full_table[1,1]/sum(fit_full_table[1,1],fit_full_table[1,2])
  fit_full_sen = fit_full_table[2,2]/sum(fit_full_table[2,1],fit_full_table[2,2])
  fit_full_acc = 1-sum(fit_full_table[1,2],fit_full_table[2,1])/sum(fit_full_table)
}

fit_minerr_table
fit_best_pruned_table
fit_full_table

#############################################################################################
#############################################################################################

# Partition index
set.seed(1)
n.train = floor(nrow(dat.ballot)*pratio)
ind.train = sample(1:nrow(dat.ballot), n.train)
ind.test = setdiff(1:nrow(dat.ballot), ind.train)

#############################################################################################
#############################################################################################
## 1. Population, Income, Church, Economy, Race, Gender, and Age predictors

PICERGA_dat = dat.ballot[,c('Voted_for_or_Not', 'Population','Per_Capita_Income',
                 'Medium_Family_Income',	'No_of_Churches',
                 'No_of_Church_Members','Poverty_Level_Rate','Unemployment_Rate',
                 'Percent_White', 'Percent_Black', 'Percent_Other', 'Percent_Minority',
                 'Percent_Male', 'Percent_Female',
                 'Age_Less_than_18', 'Age_18_24', 'Age_24_44', 
                 'Age_44_64', 'Age_Older_than_64')]
# Partition data
PICERGA_train = PICERGA_dat[ind.train,]
PICERGA_test = PICERGA_dat[ind.test,]

##########################
## PICERGA - Min. Error Tree
PICERGA_fit_minerr = rpart(Voted_for_or_Not ~ .,
                           method="class", data=PICERGA_train, cp=CPP, minsplit=MS)
PICERGA_bestcp = PICERGA_fit_minerr$cptable[which.min(PICERGA_fit_minerr$cptable[,"xerror"]),"CP"]
PICERGA_tree_minerr = prune(PICERGA_fit_minerr, cp = PICERGA_bestcp)
## plot the Min. Error Tree 
rpart.plot(PICERGA_tree_minerr, main = paste('The Min. Error Tree for PICERGA Predictors'))
# prediction by the Min. Error Tree
PICERGA_yhat_minerr = predict(PICERGA_tree_minerr, PICERGA_test, type = "prob")[,2]
# set compare with cut-off value
PICERGA_yhat_minerr_cutoff = as.numeric(PICERGA_yhat_minerr > cutoff)
# error of the Min. Error Tree
PICERGA_err_minerr = 1 - mean(PICERGA_yhat_minerr_cutoff == PICERGA_test$Voted_for_or_Not)
# sensitifity and sepcificity
PICERGA_minerr_table = table(PICERGA_test$Voted_for_or_Not, PICERGA_yhat_minerr_cutoff)
if (ncol(PICERGA_minerr_table) == 1) {
  PICERGA_minerr_spe = PICERGA_minerr_table[1,1]/sum(PICERGA_minerr_table[1,1],0)
  PICERGA_minerr_sen = 0/sum(PICERGA_minerr_table[2,1],0)
  PICERGA_minerr_acc = 1-sum(0,PICERGA_minerr_table[2,1])/sum(PICERGA_minerr_table)
} else {
  PICERGA_minerr_spe = PICERGA_minerr_table[1,1]/sum(PICERGA_minerr_table[1,1],PICERGA_minerr_table[1,2])
  PICERGA_minerr_sen = PICERGA_minerr_table[2,2]/sum(PICERGA_minerr_table[2,1],PICERGA_minerr_table[2,2])
  PICERGA_minerr_acc = 1-sum(PICERGA_minerr_table[1,2],PICERGA_minerr_table[2,1])/sum(PICERGA_minerr_table)
}

##########################
## PICERGA - Best Pruned Tree

# Find standard error, and match the standard error to its CP
PICERGA_mincp_SD = PICERGA_fit_minerr$cptable[which.min(PICERGA_fit_minerr$cptable[,"xerror"]),"xstd"]
PICERGA_mincp_SE = PICERGA_mincp_SD / sqrt(nrow(PICERGA_train))
PICERGA_pruned_SD = PICERGA_mincp_SD - PICERGA_mincp_SE
PICERGA_mincp_SD_less_split = PICERGA_fit_minerr$cptable[,'xstd'][1:which.min(PICERGA_fit_minerr$cptable[,"xerror"])]
PICERGA_pruned_CP = PICERGA_fit_minerr$cptable[which(abs(PICERGA_mincp_SD_less_split-PICERGA_pruned_SD)==min(abs(PICERGA_mincp_SD_less_split-PICERGA_pruned_SD))),"CP"]

# get the best pruned tree
PICERGA_best_pruned = prune(PICERGA_fit_minerr, cp = PICERGA_pruned_CP)
# plot the best pruned tree 
rpart.plot(PICERGA_best_pruned, main = paste('The Best Pruned Tree for PICERGA Predictors'))
# prediction by the best pruned tree
PICERGA_yhat_best_pruned = predict(PICERGA_best_pruned, PICERGA_test, type = "prob")[,2]
# set compare with cut-off value
PICERGA_yhat_pruned_cutoff = as.numeric(PICERGA_yhat_best_pruned > cutoff)
# error of the best pruned tree
PICERGA_err_best_pruned = 1 - mean(PICERGA_yhat_pruned_cutoff == PICERGA_test$Voted_for_or_Not)
# sensitifity and sepcificity
PICERGA_best_pruned_table = table(PICERGA_test$Voted_for_or_Not, PICERGA_yhat_pruned_cutoff)
if (ncol(PICERGA_best_pruned_table) == 1) {
  PICERGA_best_pruned_spe = PICERGA_best_pruned_table[1,1]/sum(PICERGA_best_pruned_table[1,1],0)
  PICERGA_best_pruned_sen = 0/sum(PICERGA_best_pruned_table[2,1],0)
  PICERGA_best_pruned_acc = 1-sum(0,PICERGA_best_pruned_table[2,1])/sum(PICERGA_best_pruned_table)
} else {
  PICERGA_best_pruned_spe = PICERGA_best_pruned_table[1,1]/sum(PICERGA_best_pruned_table[1,1],PICERGA_best_pruned_table[1,2])
  PICERGA_best_pruned_sen = PICERGA_best_pruned_table[2,2]/sum(PICERGA_best_pruned_table[2,1],PICERGA_best_pruned_table[2,2])
  PICERGA_best_pruned_acc = 1-sum(PICERGA_best_pruned_table[1,2],PICERGA_best_pruned_table[2,1])/sum(PICERGA_best_pruned_table)
}

##################
# PICERGA - full tree
PICERGA_fit_full = rpart(Voted_for_or_Not ~ ., 
                data = PICERGA_train,method="class", cp = CPP, minsplit=MS)
## plot the full tree 
rpart.plot(PICERGA_fit_full, main = paste('The full tree for PICERGA Predictors'))
# prediction by the full tree
PICERGA_yhat_full_class = predict(PICERGA_fit_full, PICERGA_test, type = "class")
PICERGA_yhat_full = predict(PICERGA_fit_full, PICERGA_test, type = "prob")[,2]
# set compare with cut-off value
PICERGA_yhat_full_cutoff = as.numeric(PICERGA_yhat_full > cutoff)
# error of the full tree
PICERGA_err_full = 1 - mean(PICERGA_yhat_full_cutoff == PICERGA_test$Voted_for_or_Not)
# sensitifity and sepcificity
PICERGA_full_table = table(PICERGA_test$Voted_for_or_Not, PICERGA_yhat_full_cutoff)
if (ncol(PICERGA_full_table) == 1) {
  PICERGA_full_spe = PICERGA_full_table[1,1]/sum(PICERGA_full_table[1,1],0)
  PICERGA_full_sen = 0/sum(PICERGA_full_table[2,1],0)
  PICERGA_full_acc = 1-sum(0,PICERGA_full_table[2,1])/sum(PICERGA_full_table)
} else {
  PICERGA_full_spe = PICERGA_full_table[1,1]/sum(PICERGA_full_table[1,1],PICERGA_full_table[1,2])
  PICERGA_full_sen = PICERGA_full_table[2,2]/sum(PICERGA_full_table[2,1],PICERGA_full_table[2,2])
  PICERGA_full_acc = 1-sum(PICERGA_full_table[1,2],PICERGA_full_table[2,1])/sum(PICERGA_full_table)
}

#############################################################################################
#############################################################################################
## 2. Income

INC_dat = dat.ballot[,c('Voted_for_or_Not', 'Per_Capita_Income', 'Medium_Family_Income')]
# Partition data
INC_train = INC_dat[ind.train,]
INC_test = INC_dat[ind.test,]

##########################
## Income - Min. Error Tree
INC_fit_minerr = rpart(Voted_for_or_Not ~ .,
                        method="class", data = INC_train, cp=CPP, minsplit=MS)
INC_bestcp = INC_fit_minerr$cptable[which.min(INC_fit_minerr$cptable[,"xerror"]),"CP"]
INC_tree_minerr = prune(INC_fit_minerr, cp = INC_bestcp)
## plot the Min. Error Tree 
rpart.plot(INC_tree_minerr, main = paste('The Min. Error Tree for Income Predictors'))
# prediction by the Min. Error Tree
INC_yhat_minerr = predict(INC_tree_minerr, INC_test, type = "prob")[,2]
# set compare with cut-off value
INC_yhat_minerr_cutoff = as.numeric(INC_yhat_minerr > cutoff)
# error of the Min. Error Tree
INC_err_minerr = 1 - mean(INC_yhat_minerr_cutoff == INC_test$Voted_for_or_Not)
# sensitifity and sepcificity
INC_minerr_table = table(INC_test$Voted_for_or_Not, INC_yhat_minerr_cutoff)
if (ncol(INC_minerr_table) == 1) {
  INC_minerr_spe = INC_minerr_table[1,1]/sum(INC_minerr_table[1,1],0)
  INC_minerr_sen = 0/sum(INC_minerr_table[2,1],0)
  INC_minerr_acc = 1-sum(0,INC_minerr_table[2,1])/sum(INC_minerr_table)
} else {
  INC_minerr_spe = INC_minerr_table[1,1]/sum(INC_minerr_table[1,1],INC_minerr_table[1,2])
  INC_minerr_sen = INC_minerr_table[2,2]/sum(INC_minerr_table[2,1],INC_minerr_table[2,2])
  INC_minerr_acc = 1-sum(INC_minerr_table[1,2],INC_minerr_table[2,1])/sum(INC_minerr_table)
}

##########################
## Income - Best Pruned Tree

# Find standard error, and match the standard error to its CP
INC_mincp_SD = INC_fit_minerr$cptable[which.min(INC_fit_minerr$cptable[,"xerror"]),"xstd"]
INC_mincp_SE = INC_mincp_SD / sqrt(nrow(INC_train))
INC_pruned_SD = INC_mincp_SD - INC_mincp_SE
INC_mincp_SD_less_split = INC_fit_minerr$cptable[,'xstd'][1:which.min(INC_fit_minerr$cptable[,"xerror"])]
INC_pruned_CP = INC_fit_minerr$cptable[which(abs(INC_mincp_SD_less_split-INC_pruned_SD)==min(abs(INC_mincp_SD_less_split-INC_pruned_SD))),"CP"]

# get the best pruned tree
INC_best_pruned = prune(INC_fit_minerr, cp = INC_pruned_CP)
# plot the Best Pruned Tree 
rpart.plot(INC_best_pruned, main = paste('The Best Pruned Tree for INC Predictors'))
# prediction by the Best Pruned Tree
INC_yhat_pruned = predict(INC_best_pruned, INC_test, type = "prob")[,2]
# set compare with cut-off value
INC_yhat_pruned_cutoff = as.numeric(INC_yhat_pruned > cutoff)
# error of the best pruned tree
INC_err_best_pruned = 1 - mean(INC_yhat_pruned_cutoff == INC_test$Voted_for_or_Not)
# sensitifity and sepcificity
INC_best_pruned_table = table(INC_test$Voted_for_or_Not, INC_yhat_pruned_cutoff)
if (ncol(INC_best_pruned_table) == 1) {
  INC_best_pruned_spe = INC_best_pruned_table[1,1]/sum(INC_best_pruned_table[1,1],0)
  INC_best_pruned_sen = 0/sum(INC_best_pruned_table[2,1],0)
  INC_best_pruned_acc = 1-sum(0,INC_best_pruned_table[2,1])/sum(INC_best_pruned_table)
} else {
  INC_best_pruned_spe = INC_best_pruned_table[1,1]/sum(INC_best_pruned_table[1,1],INC_best_pruned_table[1,2])
  INC_best_pruned_sen = INC_best_pruned_table[2,2]/sum(INC_best_pruned_table[2,1],INC_best_pruned_table[2,2])
  INC_best_pruned_acc = 1-sum(INC_best_pruned_table[1,2],INC_best_pruned_table[2,1])/sum(INC_best_pruned_table)
}

##################
# Income - full tree
INC_fit_full <- rpart(Voted_for_or_Not ~ ., 
                       data = INC_train,method = "class",cp = CPP, minsplit=MS)
## plot the full tree 
rpart.plot(INC_fit_full, main = paste('The full tree for Income Predictors'))
# prediction by the full tree
INC_yhat_full = predict(INC_fit_full, INC_test, type = "prob")[,2]
# set compare with cut-off value
INC_yhat_full_cutoff = as.numeric(INC_yhat_full > cutoff)
# error of the full tree
INC_err_full = 1 - mean(INC_yhat_full_cutoff == INC_test$Voted_for_or_Not)
# sensitifity and sepcificity
INC_full_table = table(INC_test$Voted_for_or_Not, INC_yhat_full_cutoff)
if (ncol(INC_full_table) == 1) {
  INC_full_spe = INC_full_table[1,1]/sum(INC_full_table[1,1],0)
  INC_full_sen = 0/sum(INC_full_table[2,1],0)
  INC_full_acc = 1-sum(0,INC_full_table[2,1])/sum(INC_full_table)
} else {
  INC_full_spe = INC_full_table[1,1]/sum(INC_full_table[1,1],INC_full_table[1,2])
  INC_full_sen = INC_full_table[2,2]/sum(INC_full_table[2,1],INC_full_table[2,2])
  INC_full_acc = 1-sum(INC_full_table[1,2],INC_full_table[2,1])/sum(INC_full_table)
}

#############################################################################################
#############################################################################################
## 3. Church predictors

CHUR_dat = dat.ballot[,c('Voted_for_or_Not', 'No_of_Churches', 'No_of_Church_Members')]
# Partition data
CHUR_train = CHUR_dat[ind.train,]
CHUR_test = CHUR_dat[ind.test,]

#############################
## Churche - Min. Error Tree
CHUR_fit_minerr = rpart(Voted_for_or_Not ~ .,
                       method="class", data = CHUR_train, cp=CPP, minsplit=MS)
CHUR_bestcp = CHUR_fit_minerr$cptable[which.min(CHUR_fit_minerr$cptable[,"xerror"]),"CP"]
CHUR_tree_minerr = prune(CHUR_fit_minerr, cp = CHUR_bestcp)
## plot the Min. Error Tree 
rpart.plot(CHUR_tree_minerr, main = paste('The Min. Error Tree for Churche Predictors'))
# prediction by the Min. Error Tree
CHUR_yhat_minerr = predict(CHUR_tree_minerr, CHUR_test, type = "prob")[,2]
# set compare with cut-off value
CHUR_yhat_minerr_cutoff = as.numeric(CHUR_yhat_minerr > cutoff)
# error of the Min. Error Tree
CHUR_err_minerr = 1 - mean(CHUR_yhat_minerr_cutoff == CHUR_test$Voted_for_or_Not)
# sensitifity and sepcificity
CHUR_minerr_table = table(CHUR_test$Voted_for_or_Not, CHUR_yhat_minerr_cutoff)
if (ncol(CHUR_minerr_table) == 1) {
  CHUR_minerr_spe = CHUR_minerr_table[1,1]/sum(CHUR_minerr_table[1,1],0)
  CHUR_minerr_sen = 0/sum(CHUR_minerr_table[2,1],0)
  CHUR_minerr_acc = 1-sum(0,CHUR_minerr_table[2,1])/sum(CHUR_minerr_table)
} else {
  CHUR_minerr_spe = CHUR_minerr_table[1,1]/sum(CHUR_minerr_table[1,1],CHUR_minerr_table[1,2])
  CHUR_minerr_sen = CHUR_minerr_table[2,2]/sum(CHUR_minerr_table[2,1],CHUR_minerr_table[2,2])
  CHUR_minerr_acc = 1-sum(CHUR_minerr_table[1,2],CHUR_minerr_table[2,1])/sum(CHUR_minerr_table)
}

##########################
## Church - Best Pruned Tree

# Find standard error, and match the standard error to its CP
CHUR_mincp_SD = CHUR_fit_minerr$cptable[which.min(CHUR_fit_minerr$cptable[,"xerror"]),"xstd"]
CHUR_mincp_SE = CHUR_mincp_SD / sqrt(nrow(CHUR_train))
CHUR_pruned_SD = CHUR_mincp_SD - CHUR_mincp_SE
CHUR_mincp_SD_less_split = CHUR_fit_minerr$cptable[,'xstd'][1:which.min(CHUR_fit_minerr$cptable[,"xerror"])]
CHUR_pruned_CP = CHUR_fit_minerr$cptable[which(abs(CHUR_mincp_SD_less_split-CHUR_pruned_SD)==min(abs(CHUR_mincp_SD_less_split-CHUR_pruned_SD))),"CP"]

# get the best pruned tree
CHUR_best_pruned = prune(CHUR_fit_minerr, cp = CHUR_pruned_CP)
# plot the Best Pruned Tree 
rpart.plot(CHUR_best_pruned, main = paste('The Best Pruned Tree for Church Predictors'))
# prediction by the Best Pruned Tree
CHUR_yhat_pruned = predict(CHUR_best_pruned, CHUR_test, type = "prob")[,2]
# set compare with cut-off value
CHUR_yhat_pruned_cutoff = as.numeric(CHUR_yhat_pruned > cutoff)
# error of the best pruned tree
CHUR_err_best_pruned = 1 - mean(CHUR_yhat_pruned_cutoff == CHUR_test$Voted_for_or_Not)
# sensitifity and sepcificity
CHUR_best_pruned_table = table(CHUR_test$Voted_for_or_Not, CHUR_yhat_pruned_cutoff)
if (ncol(CHUR_best_pruned_table) == 1) {
  CHUR_best_pruned_spe = CHUR_best_pruned_table[1,1]/sum(CHUR_best_pruned_table[1,1],0)
  CHUR_best_pruned_sen = 0/sum(CHUR_best_pruned_table[2,1],0)
  CHUR_best_pruned_acc = 1-sum(0,CHUR_best_pruned_table[2,1])/sum(CHUR_best_pruned_table)
} else {
  CHUR_best_pruned_spe = CHUR_best_pruned_table[1,1]/sum(CHUR_best_pruned_table[1,1],CHUR_best_pruned_table[1,2])
  CHUR_best_pruned_sen = CHUR_best_pruned_table[2,2]/sum(CHUR_best_pruned_table[2,1],CHUR_best_pruned_table[2,2])
  CHUR_best_pruned_acc = 1-sum(CHUR_best_pruned_table[1,2],CHUR_best_pruned_table[2,1])/sum(CHUR_best_pruned_table)
}

######################
# Churche - full tree
CHUR_fit_full <- rpart(Voted_for_or_Not ~ ., 
                      data = CHUR_train,method = "class", cp = CPP, minsplit=MS)
## plot the full tree 
rpart.plot(CHUR_fit_full, main = paste('The full tree for Churche Predictors'))
# prediction by the full tree
CHUR_yhat_full = predict(CHUR_fit_full, CHUR_test, type = "prob")[,2]
# set compare with cut-off value
CHUR_yhat_full_cutoff = as.numeric(CHUR_yhat_full > cutoff)
# error of the full tree
CHUR_err_full = 1 - mean(CHUR_yhat_full_cutoff == CHUR_test$Voted_for_or_Not)
# sensitifity and sepcificity
CHUR_full_table = table(CHUR_test$Voted_for_or_Not, CHUR_yhat_full_cutoff)
if (ncol(CHUR_full_table) == 1) {
  CHUR_full_spe = CHUR_full_table[1,1]/sum(CHUR_full_table[1,1],0)
  CHUR_full_sen = 0/sum(CHUR_full_table[2,1],0)
  CHUR_full_acc = 1-sum(0,CHUR_full_table[2,1])/sum(CHUR_full_table)
} else {
  CHUR_full_spe = CHUR_full_table[1,1]/sum(CHUR_full_table[1,1],CHUR_full_table[1,2])
  CHUR_full_sen = CHUR_full_table[2,2]/sum(CHUR_full_table[2,1],CHUR_full_table[2,2])
  CHUR_full_acc = 1-sum(CHUR_full_table[1,2],CHUR_full_table[2,1])/sum(CHUR_full_table)
}

#############################################################################################
#############################################################################################
## 4. Economy predictors

ECON_dat = dat.ballot[,c('Voted_for_or_Not', 'Poverty_Level_Rate','Unemployment_Rate')]
# Partition data
ECON_train = ECON_dat[ind.train,]
ECON_test = ECON_dat[ind.test,]

##########################
## Economy - Min. Error Tree
ECON_fit_minerr = rpart(Voted_for_or_Not ~ .,
                        method="class", data = ECON_train, cp=CPP, minsplit=MS)
ECON_bestcp = ECON_fit_minerr$cptable[which.min(ECON_fit_minerr$cptable[,"xerror"]),"CP"]
ECON_tree_minerr = prune(ECON_fit_minerr, cp = ECON_bestcp)
## plot the Min. Error Tree 
rpart.plot(ECON_tree_minerr, main = paste('The Min. Error Tree for Economy Predictors'))
# prediction by the Min. Error Tree
ECON_yhat_minerr = predict(ECON_tree_minerr, ECON_test, type = "prob")[,2]
# set compare with cut-off value
ECON_yhat_minerr_cutoff = as.numeric(ECON_yhat_minerr > cutoff)
# error of the Min. Error Tree
ECON_err_minerr = 1 - mean(ECON_yhat_minerr_cutoff == ECON_test$Voted_for_or_Not)
# sensitifity and sepcificity
ECON_minerr_table = table(ECON_test$Voted_for_or_Not, ECON_yhat_minerr_cutoff)
if (ncol(ECON_minerr_table) == 1) {
  ECON_minerr_spe = ECON_minerr_table[1,1]/sum(ECON_minerr_table[1,1],0)
  ECON_minerr_sen = 0/sum(ECON_minerr_table[2,1],0)
  ECON_minerr_acc = 1-sum(0,ECON_minerr_table[2,1])/sum(ECON_minerr_table)
} else {
  ECON_minerr_spe = ECON_minerr_table[1,1]/sum(ECON_minerr_table[1,1],ECON_minerr_table[1,2])
  ECON_minerr_sen = ECON_minerr_table[2,2]/sum(ECON_minerr_table[2,1],ECON_minerr_table[2,2])
  ECON_minerr_acc = 1-sum(ECON_minerr_table[1,2],ECON_minerr_table[2,1])/sum(ECON_minerr_table)
}

##########################
## Economy - Best Pruned Tree

# Find standard error, and match the standard error to its CP
ECON_mincp_SD = ECON_fit_minerr$cptable[which.min(ECON_fit_minerr$cptable[,"xerror"]),"xstd"]
ECON_mincp_SE = ECON_mincp_SD / sqrt(nrow(ECON_train))
ECON_pruned_SD = ECON_mincp_SD - ECON_mincp_SE
ECON_mincp_SD_less_split = ECON_fit_minerr$cptable[,'xstd'][1:which.min(ECON_fit_minerr$cptable[,"xerror"])]
ECON_pruned_CP = ECON_fit_minerr$cptable[which(abs(ECON_mincp_SD_less_split-ECON_pruned_SD)==min(abs(ECON_mincp_SD_less_split-ECON_pruned_SD))),"CP"]

# get the best pruned tree
ECON_best_pruned = prune(ECON_fit_minerr, cp = ECON_pruned_CP)
# plot the Best Pruned Tree 
rpart.plot(ECON_best_pruned, main = paste('The Best Pruned Tree for Economy Predictors'))
# prediction by the Best Pruned Tree
ECON_yhat_pruned = predict(ECON_best_pruned, ECON_test, type = "prob")[,2]
# set compare with cut-off value
ECON_yhat_pruned_cutoff = as.numeric(ECON_yhat_pruned > cutoff)
# error of the best pruned tree
ECON_err_best_pruned = 1 - mean(ECON_yhat_pruned_cutoff == ECON_test$Voted_for_or_Not)
# sensitifity and sepcificity
ECON_best_pruned_table = table(ECON_test$Voted_for_or_Not, ECON_yhat_pruned_cutoff)
if (ncol(ECON_best_pruned_table) == 1) {
  ECON_best_pruned_spe = ECON_best_pruned_table[1,1]/sum(ECON_best_pruned_table[1,1],0)
  ECON_best_pruned_sen = 0/sum(ECON_best_pruned_table[2,1],0)
  ECON_best_pruned_acc = 1-sum(0,ECON_best_pruned_table[2,1])/sum(ECON_best_pruned_table)
} else {
  ECON_best_pruned_spe = ECON_best_pruned_table[1,1]/sum(ECON_best_pruned_table[1,1],ECON_best_pruned_table[1,2])
  ECON_best_pruned_sen = ECON_best_pruned_table[2,2]/sum(ECON_best_pruned_table[2,1],ECON_best_pruned_table[2,2])
  ECON_best_pruned_acc = 1-sum(ECON_best_pruned_table[1,2],ECON_best_pruned_table[2,1])/sum(ECON_best_pruned_table)
}

######################
# Economy - full tree
ECON_fit_full <- rpart(Voted_for_or_Not ~ ., 
                       data = ECON_train,method = "class", cp = CPP, minsplit=MS)
## plot the full tree 
rpart.plot(ECON_fit_full, main = paste('The full tree for Economy Predictors'))
# prediction by the full tree
ECON_yhat_full = predict(ECON_fit_full, ECON_test, type = "prob")[,2]
# set compare with cut-off value
ECON_yhat_full_cutoff = as.numeric(ECON_yhat_full > cutoff)
# error of the full tree
ECON_err_full = 1 - mean(ECON_yhat_full_cutoff == ECON_test$Voted_for_or_Not)
# sensitifity and sepcificity
ECON_full_table = table(ECON_test$Voted_for_or_Not, ECON_yhat_full_cutoff)
if (ncol(ECON_full_table) == 1) {
  ECON_full_spe = ECON_full_table[1,1]/sum(ECON_full_table[1,1],0)
  ECON_full_sen = 0/sum(ECON_full_table[2,1],0)
  ECON_full_acc = 1-sum(0,ECON_full_table[2,1])/sum(ECON_full_table)
} else {
  ECON_full_spe = ECON_full_table[1,1]/sum(ECON_full_table[1,1],ECON_full_table[1,2])
  ECON_full_sen = ECON_full_table[2,2]/sum(ECON_full_table[2,1],ECON_full_table[2,2])
  ECON_full_acc = 1-sum(ECON_full_table[1,2],ECON_full_table[2,1])/sum(ECON_full_table)
}

#############################################################################################
#############################################################################################
## 5. Race predictors

RACE_dat = dat.ballot[,c('Voted_for_or_Not', 'Percent_White', 'Percent_Black', 'Percent_Other', 
                        'Percent_Minority')]
# Partition data
RACE_train = RACE_dat[ind.train,]
RACE_test = RACE_dat[ind.test,]

##########################
## Race - Min. Error Tree
RACE_fit_minerr = rpart(Voted_for_or_Not ~ ., method="class", data = RACE_train, cp=CPP, minsplit=MS)
RACE_bestcp = RACE_fit_minerr$cptable[which.min(RACE_fit_minerr$cptable[,"xerror"]),"CP"]
RACE_tree_minerr = prune(RACE_fit_minerr, cp = RACE_bestcp)
## plot the Min. Error Tree 
rpart.plot(RACE_tree_minerr, main = paste('The Min. Error Tree for Race Predictors'))
# prediction by the Min. Error Tree
RACE_yhat_minerr = predict(RACE_tree_minerr, RACE_test, type = "prob")[,2]
# set compare with cut-off value
RACE_yhat_minerr_cutoff = as.numeric(RACE_yhat_minerr > cutoff)
# error of the Min. Error Tree
RACE_err_minerr = 1 - mean(RACE_yhat_minerr_cutoff == RACE_test$Voted_for_or_Not)
# sensitifity and sepcificity
RACE_minerr_table = table(RACE_test$Voted_for_or_Not, RACE_yhat_minerr_cutoff)
if (ncol(RACE_minerr_table) == 1) {
  RACE_minerr_spe = RACE_minerr_table[1,1]/sum(RACE_minerr_table[1,1],0)
  RACE_minerr_sen = 0/sum(RACE_minerr_table[2,1],0)
  RACE_minerr_acc = 1-sum(0,RACE_minerr_table[2,1])/sum(RACE_minerr_table)
} else {
  RACE_minerr_spe = RACE_minerr_table[1,1]/sum(RACE_minerr_table[1,1],RACE_minerr_table[1,2])
  RACE_minerr_sen = RACE_minerr_table[2,2]/sum(RACE_minerr_table[2,1],RACE_minerr_table[2,2])
  RACE_minerr_acc = 1-sum(RACE_minerr_table[1,2],RACE_minerr_table[2,1])/sum(RACE_minerr_table)
}

##########################
## Race - Best Pruned Tree

# Find standard error, and match the standard error to its CP
RACE_mincp_SD = RACE_fit_minerr$cptable[which.min(RACE_fit_minerr$cptable[,"xerror"]),"xstd"]
RACE_mincp_SE = RACE_mincp_SD / sqrt(nrow(RACE_train))
RACE_pruned_SD = RACE_mincp_SD - RACE_mincp_SE
RACE_mincp_SD_less_split = RACE_fit_minerr$cptable[,'xstd'][1:which.min(RACE_fit_minerr$cptable[,"xerror"])]
RACE_pruned_CP = RACE_fit_minerr$cptable[which(abs(RACE_mincp_SD_less_split-RACE_pruned_SD)==min(abs(RACE_mincp_SD_less_split-RACE_pruned_SD))),"CP"]

# get the best pruned tree
RACE_best_pruned = prune(RACE_fit_minerr, cp = RACE_pruned_CP)
# plot the Best Pruned Tree 
rpart.plot(RACE_best_pruned, main = paste('The Best Pruned Tree for RACE Predictors'))
# prediction by the Best Pruned Tree
RACE_yhat_pruned = predict(RACE_best_pruned, RACE_test, type = "prob")[,2]
# set compare with cut-off value
RACE_yhat_pruned_cutoff = as.numeric(RACE_yhat_pruned > cutoff)
# error of the best pruned tree
RACE_err_best_pruned = 1 - mean(RACE_yhat_pruned_cutoff == RACE_test$Voted_for_or_Not)
# sensitifity and sepcificity
RACE_best_pruned_table = table(RACE_test$Voted_for_or_Not, RACE_yhat_pruned_cutoff)
if (ncol(RACE_best_pruned_table) == 1) {
  RACE_best_pruned_spe = RACE_best_pruned_table[1,1]/sum(RACE_best_pruned_table[1,1],0)
  RACE_best_pruned_sen = 0/sum(RACE_best_pruned_table[2,1],0)
  RACE_best_pruned_acc = 1-sum(0,RACE_best_pruned_table[2,1])/sum(RACE_best_pruned_table)
} else {
  RACE_best_pruned_spe = RACE_best_pruned_table[1,1]/sum(RACE_best_pruned_table[1,1],RACE_best_pruned_table[1,2])
  RACE_best_pruned_sen = RACE_best_pruned_table[2,2]/sum(RACE_best_pruned_table[2,1],RACE_best_pruned_table[2,2])
  RACE_best_pruned_acc = 1-sum(RACE_best_pruned_table[1,2],RACE_best_pruned_table[2,1])/sum(RACE_best_pruned_table)
}

##################
# Race - full tree
RACE_fit_full <- rpart(Voted_for_or_Not ~ ., data = RACE_train,method = "class", cp = CPP, minsplit=MS)
## plot the full tree 
rpart.plot(RACE_fit_full, main = paste('The full tree for Race Predictors'))
# prediction by the full tree
RACE_yhat_full = predict(RACE_fit_full, RACE_test, type = "prob")[,2]
# set compare with cut-off value
RACE_yhat_full_cutoff = as.numeric(RACE_yhat_full > cutoff)
# error of the full tree
RACE_err_full = 1 - mean(RACE_yhat_full_cutoff == RACE_test$Voted_for_or_Not)
# sensitifity and sepcificity
RACE_full_table = table(RACE_test$Voted_for_or_Not, RACE_yhat_full_cutoff)
if (ncol(RACE_full_table) == 1) {
  RACE_full_spe = RACE_full_table[1,1]/sum(RACE_full_table[1,1],0)
  RACE_full_sen = 0/sum(RACE_full_table[2,1],0)
  RACE_full_acc = 1-sum(0,RACE_full_table[2,1])/sum(RACE_full_table)
} else {
  RACE_full_spe = RACE_full_table[1,1]/sum(RACE_full_table[1,1],RACE_full_table[1,2])
  RACE_full_sen = RACE_full_table[2,2]/sum(RACE_full_table[2,1],RACE_full_table[2,2])
  RACE_full_acc = 1-sum(RACE_full_table[1,2],RACE_full_table[2,1])/sum(RACE_full_table)
}

#############################################################################################
#############################################################################################
## 6. Age predictors

Age_dat = dat.ballot[,c('Voted_for_or_Not', 'Age_Less_than_18', 'Age_18_24', 'Age_24_44', 
                         'Age_44_64', 'Age_Older_than_64')]
# Partition data
Age_train = Age_dat[ind.train,]
Age_test = Age_dat[ind.test,]

##########################
## Age - Min. Error Tree
Age_fit_minerr = rpart(Voted_for_or_Not ~ ., method="class", data = Age_train, cp=CPP, minsplit=MS)
Age_bestcp = Age_fit_minerr$cptable[which.min(Age_fit_minerr$cptable[,"xerror"]),"CP"]
Age_tree_minerr = prune(Age_fit_minerr, cp = Age_bestcp)
## plot the Min. Error Tree 
rpart.plot(Age_tree_minerr, main = paste('The Min. Error Tree for Age Predictors'))
# prediction by the Min. Error Tree
Age_yhat_minerr = predict(Age_tree_minerr, Age_test, type = "prob")[,2]
# set compare with cut-off value
Age_yhat_minerr_cutoff = as.numeric(Age_yhat_minerr > cutoff)
# error of the Min. Error Tree
Age_err_minerr = 1 - mean(Age_yhat_minerr_cutoff == Age_test$Voted_for_or_Not)
# sensitifity and sepcificity
Age_minerr_table = table(Age_test$Voted_for_or_Not, Age_yhat_minerr_cutoff)
if (ncol(Age_minerr_table) == 1) {
  Age_minerr_spe = Age_minerr_table[1,1]/sum(Age_minerr_table[1,1],0)
  Age_minerr_sen = 0/sum(Age_minerr_table[2,1],0)
  Age_minerr_acc = 1-sum(0,Age_minerr_table[2,1])/sum(Age_minerr_table)
} else {
  Age_minerr_spe = Age_minerr_table[1,1]/sum(Age_minerr_table[1,1],Age_minerr_table[1,2])
  Age_minerr_sen = Age_minerr_table[2,2]/sum(Age_minerr_table[2,1],Age_minerr_table[2,2])
  Age_minerr_acc = 1-sum(Age_minerr_table[1,2],Age_minerr_table[2,1])/sum(Age_minerr_table)
}

##########################
## Age - Best Pruned Tree

# Find standard error, and match the standard error to its CP
Age_mincp_SD = Age_fit_minerr$cptable[which.min(Age_fit_minerr$cptable[,"xerror"]),"xstd"]
Age_mincp_SE = Age_mincp_SD / sqrt(nrow(Age_train))
Age_pruned_SD = Age_mincp_SD - Age_mincp_SE
Age_mincp_SD_less_split = Age_fit_minerr$cptable[,'xstd'][1:which.min(Age_fit_minerr$cptable[,"xerror"])]
Age_pruned_CP = Age_fit_minerr$cptable[which(abs(Age_mincp_SD_less_split-Age_pruned_SD)==min(abs(Age_mincp_SD_less_split-Age_pruned_SD))),"CP"]

# get the best pruned tree
Age_best_pruned = prune(Age_fit_minerr, cp = Age_pruned_CP)
# plot the Best Pruned Tree 
rpart.plot(Age_best_pruned, main = paste('The Best Pruned Tree for Age Predictors'))
# prediction by the Best Pruned Tree
Age_yhat_pruned = predict(Age_best_pruned, Age_test, type = "prob")[,2]
# set compare with cut-off value
Age_yhat_pruned_cutoff = as.numeric(Age_yhat_pruned > cutoff)
# error of the best pruned tree
Age_err_best_pruned = 1 - mean(Age_yhat_pruned_cutoff == Age_test$Voted_for_or_Not)
# sensitifity and sepcificity
Age_best_pruned_table = table(Age_test$Voted_for_or_Not, Age_yhat_pruned_cutoff)
if (ncol(Age_best_pruned_table) == 1) {
  Age_best_pruned_spe = Age_best_pruned_table[1,1]/sum(Age_best_pruned_table[1,1],0)
  Age_best_pruned_sen = 0/sum(Age_best_pruned_table[2,1],0)
  Age_best_pruned_acc = 1-sum(0,Age_best_pruned_table[2,1])/sum(Age_best_pruned_table)
} else {
  Age_best_pruned_spe = Age_best_pruned_table[1,1]/sum(Age_best_pruned_table[1,1],Age_best_pruned_table[1,2])
  Age_best_pruned_sen = Age_best_pruned_table[2,2]/sum(Age_best_pruned_table[2,1],Age_best_pruned_table[2,2])
  Age_best_pruned_acc = 1-sum(Age_best_pruned_table[1,2],Age_best_pruned_table[2,1])/sum(Age_best_pruned_table)
}

##################
# Age - full tree
Age_fit_full <- rpart(Voted_for_or_Not ~ ., data = Age_train,method = "class", cp = CPP, minsplit=MS)
## plot the full tree 
rpart.plot(Age_fit_full, main = paste('The full tree for Age Predictors'))
# prediction by the full tree
Age_yhat_full = predict(Age_fit_full, Age_test, type = "prob")[,2]
# set compare with cut-off value
Age_yhat_full_cutoff = as.numeric(Age_yhat_full > cutoff)
# error of the full tree
Age_err_full = 1 - mean(Age_yhat_full_cutoff == Age_test$Voted_for_or_Not)
# sensitifity and sepcificity
Age_full_table = table(Age_test$Voted_for_or_Not, Age_yhat_full_cutoff)
if (ncol(Age_full_table) == 1) {
  Age_full_spe = Age_full_table[1,1]/sum(Age_full_table[1,1],0)
  Age_full_sen = 0/sum(Age_full_table[2,1],0)
  Age_full_acc = 1-sum(0,Age_full_table[2,1])/sum(Age_full_table)
} else {
  Age_full_spe = Age_full_table[1,1]/sum(Age_full_table[1,1],Age_full_table[1,2])
  Age_full_sen = Age_full_table[2,2]/sum(Age_full_table[2,1],Age_full_table[2,2])
  Age_full_acc = 1-sum(Age_full_table[1,2],Age_full_table[2,1])/sum(Age_full_table)
}

#############################################################################################
#############################################################################################
## 7. Gender predictors

GEN_dat = dat.ballot[,c('Voted_for_or_Not', 'Percent_Male', 'Percent_Female')]
# Partition data
GEN_train = GEN_dat[ind.train,]
GEN_test = GEN_dat[ind.test,]

##########################
## Gender - Min. Error Tree
GEN_fit_minerr = rpart(Voted_for_or_Not ~ ., method="class", data = GEN_train, cp=CPP, minsplit=MS)
GEN_bestcp = GEN_fit_minerr$cptable[which.min(GEN_fit_minerr$cptable[,"xerror"]),"CP"]
GEN_tree_minerr = prune(GEN_fit_minerr, cp = GEN_bestcp)
## plot the Min. Error Tree 
rpart.plot(GEN_tree_minerr, main = paste('The Min. Error Tree for Gender Predictors'))
# prediction by the Min. Error Tree
GEN_yhat_minerr = predict(GEN_tree_minerr, GEN_test, type = "prob")[,2]
# set compare with cut-off value
GEN_yhat_minerr_cutoff = as.numeric(GEN_yhat_minerr > cutoff)
# error of the Min. Error Tree
GEN_err_minerr = 1 - mean(GEN_yhat_minerr_cutoff == GEN_test$Voted_for_or_Not)
# sensitifity and sepcificity
GEN_minerr_table = table(GEN_test$Voted_for_or_Not, GEN_yhat_minerr_cutoff)
if (ncol(GEN_minerr_table) == 1) {
  GEN_minerr_spe = GEN_minerr_table[1,1]/sum(GEN_minerr_table[1,1],0)
  GEN_minerr_sen = 0/sum(GEN_minerr_table[2,1],0)
  GEN_minerr_acc = 1-sum(0,GEN_minerr_table[2,1])/sum(GEN_minerr_table)
} else {
  GEN_minerr_spe = GEN_minerr_table[1,1]/sum(GEN_minerr_table[1,1],GEN_minerr_table[1,2])
  GEN_minerr_sen = GEN_minerr_table[2,2]/sum(GEN_minerr_table[2,1],GEN_minerr_table[2,2])
  GEN_minerr_acc = 1-sum(GEN_minerr_table[1,2],GEN_minerr_table[2,1])/sum(GEN_minerr_table)
}

##########################
## Gender - Best Pruned Tree

# Find standard error, and match the standard error to its CP
GEN_mincp_SD = GEN_fit_minerr$cptable[which.min(GEN_fit_minerr$cptable[,"xerror"]),"xstd"]
GEN_mincp_SE = GEN_mincp_SD / sqrt(nrow(GEN_train))
GEN_pruned_SD = GEN_mincp_SD - GEN_mincp_SE
GEN_mincp_SD_less_split = GEN_fit_minerr$cptable[,'xstd'][1:which.min(GEN_fit_minerr$cptable[,"xerror"])]
GEN_pruned_CP = GEN_fit_minerr$cptable[which(abs(GEN_mincp_SD_less_split-GEN_pruned_SD)==min(abs(GEN_mincp_SD_less_split-GEN_pruned_SD))),"CP"]

# get the best pruned tree
GEN_best_pruned = prune(GEN_fit_minerr, cp = GEN_pruned_CP)
# plot the Best Pruned Tree 
rpart.plot(GEN_best_pruned, main = paste('The Best Pruned Tree for Gender Predictors'))
# prediction by the Best Pruned Tree
GEN_yhat_pruned = predict(GEN_best_pruned, GEN_test, type = "prob")[,2]
# set compare with cut-off value
GEN_yhat_pruned_cutoff = as.numeric(GEN_yhat_pruned > cutoff)
# error of the best pruned tree
GEN_err_best_pruned = 1 - mean(GEN_yhat_pruned_cutoff == GEN_test$Voted_for_or_Not)
# sensitifity and sepcificity
GEN_best_pruned_table = table(GEN_test$Voted_for_or_Not, GEN_yhat_pruned_cutoff)
if (ncol(GEN_best_pruned_table) == 1) {
  GEN_best_pruned_spe = GEN_best_pruned_table[1,1]/sum(GEN_best_pruned_table[1,1],0)
  GEN_best_pruned_sen = 0/sum(GEN_best_pruned_table[2,1],0)
  GEN_best_pruned_acc = 1-sum(0,GEN_best_pruned_table[2,1])/sum(GEN_best_pruned_table)
} else {
  GEN_best_pruned_spe = GEN_best_pruned_table[1,1]/sum(GEN_best_pruned_table[1,1],GEN_best_pruned_table[1,2])
  GEN_best_pruned_sen = GEN_best_pruned_table[2,2]/sum(GEN_best_pruned_table[2,1],GEN_best_pruned_table[2,2])
  GEN_best_pruned_acc = 1-sum(GEN_best_pruned_table[1,2],GEN_best_pruned_table[2,1])/sum(GEN_best_pruned_table)
}

##################
# Gender - full tree
GEN_fit_full = rpart(Voted_for_or_Not ~ ., data = GEN_train,method = "class", cp = CPP, minsplit=MS)
## plot the full tree 
rpart.plot(GEN_fit_full, main = paste('The full tree for Gender Predictors'))
# prediction by the full tree
GEN_yhat_full = predict(GEN_fit_full, GEN_test, type = "prob")[,2]
# set compare with cut-off value
GEN_yhat_full_cutoff = as.numeric(GEN_yhat_full > cutoff)
# error of the full tree
GEN_err_full = 1 - mean(GEN_yhat_full_cutoff == GEN_test$Voted_for_or_Not)
GEN_full_table = table(GEN_test$Voted_for_or_Not, GEN_yhat_full_cutoff)
if (ncol(GEN_full_table) == 1) {
  GEN_full_spe = GEN_full_table[1,1]/sum(GEN_full_table[1,1],0)
  GEN_full_sen = 0/sum(GEN_full_table[2,1],0)
  GEN_full_acc = 1-sum(0,GEN_full_table[2,1])/sum(GEN_full_table)
} else {
  GEN_full_spe = GEN_full_table[1,1]/sum(GEN_full_table[1,1],GEN_full_table[1,2])
  GEN_full_sen = GEN_full_table[2,2]/sum(GEN_full_table[2,1],GEN_full_table[2,2])
  GEN_full_acc = 1-sum(GEN_full_table[1,2],GEN_full_table[2,1])/sum(GEN_full_table)
}

#############################################################################################
#############################################################################################
## Error Comparison of predictor groups

########################
# Min. Error Tree
all_min_error_minerr = matrix(1:35, nrow = 7, ncol = 5)
colnames(all_min_error_minerr) = c("Min error tree error",'Sensitivity','Specificity','Accuracy','Cut-Off')
rownames(all_min_error_minerr) = c('PICERGA', 'Income', 'Church', 'Economy', 'Race', 'Age', 'Gender')
all_min_error_minerr[,"Min error tree error"] = c(PICERGA_err_minerr,INC_err_minerr,CHUR_err_minerr,
                             ECON_err_minerr, RACE_err_minerr, Age_err_minerr,
                             GEN_err_minerr)
all_min_error_minerr[,"Sensitivity"] = c(PICERGA_minerr_sen,INC_minerr_sen,CHUR_minerr_sen,ECON_minerr_sen, 
                                         RACE_minerr_sen, Age_minerr_sen,GEN_minerr_sen)
all_min_error_minerr[,"Specificity"] = c(PICERGA_minerr_spe,INC_minerr_spe,CHUR_minerr_spe,
                                         ECON_minerr_spe, RACE_minerr_spe, Age_minerr_spe,GEN_minerr_spe)
all_min_error_minerr[,"Accuracy"] = c(PICERGA_minerr_acc,INC_minerr_acc,CHUR_minerr_acc,ECON_minerr_acc, 
                                      RACE_minerr_acc, Age_minerr_acc,GEN_minerr_acc)
all_min_error_minerr[,"Cut-Off"] = c(cutoff,cutoff,cutoff, cutoff, cutoff, cutoff, cutoff)

sorted_predictors_minerr = all_min_error_minerr[order(all_min_error_minerr[,1], decreasing = F),] %>% round(rounding)


########################
# Best Pruned Tree

all_min_error_best_pruned = matrix(1:35, nrow = 7, ncol = 5)
colnames(all_min_error_best_pruned) = c("Best Pruned error",'Sensitivity','Specificity','Accuracy','Cut-Off')
rownames(all_min_error_best_pruned) = c('PICERGA', 'Income', 'Church', 'Economy', 'Race', 'Age', 'Gender')
all_min_error_best_pruned[,"Best Pruned error"] = c(PICERGA_err_best_pruned,INC_err_best_pruned,CHUR_err_best_pruned,
                                                  ECON_err_best_pruned, RACE_err_best_pruned, Age_err_best_pruned,
                                                  GEN_err_best_pruned)
all_min_error_best_pruned[,"Sensitivity"] = c(PICERGA_best_pruned_sen,INC_best_pruned_sen,CHUR_best_pruned_sen,ECON_best_pruned_sen, 
                                         RACE_best_pruned_sen, Age_best_pruned_sen,GEN_best_pruned_sen)
all_min_error_best_pruned[,"Specificity"] = c(PICERGA_best_pruned_spe,INC_best_pruned_spe,CHUR_best_pruned_spe,
                                         ECON_best_pruned_spe, RACE_best_pruned_spe, Age_best_pruned_spe,GEN_best_pruned_spe)
all_min_error_best_pruned[,"Accuracy"] = c(PICERGA_best_pruned_acc,INC_best_pruned_acc,CHUR_best_pruned_acc,ECON_best_pruned_acc, 
                                      RACE_best_pruned_acc, Age_best_pruned_acc,GEN_best_pruned_acc)
all_min_error_best_pruned[,"Cut-Off"] = c(cutoff,cutoff,cutoff, cutoff, cutoff, cutoff, cutoff)

sorted_predictors_best_pruned = all_min_error_best_pruned[order(all_min_error_best_pruned[,1], decreasing = F),] %>% round(rounding)


########################
# Full tree

all_min_error_full = matrix(1:35, nrow = 7, ncol = 5)
colnames(all_min_error_full) = c("Full tree error",'Sensitivity','Specificity','Accuracy','Cut-Off')
rownames(all_min_error_full) = c('PICERGA', 'Income', 'Church', 'Economy', 'Race', 'Age', 'Gender')
all_min_error_full[,"Full tree error"] = c(PICERGA_err_full,INC_err_full,CHUR_err_full,
                                                       ECON_err_full, RACE_err_full, Age_err_full,
                                                       GEN_err_full)
all_min_error_full[,"Sensitivity"] = c(PICERGA_full_sen,INC_full_sen,CHUR_full_sen,ECON_full_sen, 
                                              RACE_full_sen, Age_full_sen,GEN_full_sen)
all_min_error_full[,"Specificity"] = c(PICERGA_full_spe,INC_full_spe,CHUR_full_spe,
                                              ECON_full_spe, RACE_full_spe, Age_full_spe,GEN_full_spe)
all_min_error_full[,"Accuracy"] = c(PICERGA_full_acc,INC_full_acc,CHUR_full_acc,ECON_full_acc, 
                                           RACE_full_acc, Age_full_acc,GEN_full_acc)
all_min_error_full[,"Cut-Off"] = c(cutoff,cutoff,cutoff, cutoff, cutoff, cutoff, cutoff)

sorted_predictors_full = all_min_error_full[order(all_min_error_full[,1], decreasing = F),] %>% round(rounding)

########################

# Results
sorted_predictors_minerr
sorted_predictors_best_pruned
sorted_predictors_full
# Separate Tabs to View Full Results
View(sorted_predictors_minerr)
View(sorted_predictors_best_pruned)
View(sorted_predictors_full)

#############################################################################################
## ROC Curve

PICERGA_ROC = roc.curve(PICERGA_test$Voted_for_or_Not, PICERGA_yhat_minerr_cutoff)
INC_ROC = roc.curve(INC_test$Voted_for_or_Not, INC_yhat_minerr_cutoff)
CHUR_ROC = roc.curve(CHUR_test$Voted_for_or_Not, CHUR_yhat_minerr_cutoff)
ECON_ROC = roc.curve(ECON_test$Voted_for_or_Not, ECON_yhat_minerr_cutoff)
RACE_ROC = roc.curve(RACE_test$Voted_for_or_Not, RACE_yhat_minerr_cutoff)
Age_ROC = roc.curve(Age_test$Voted_for_or_Not, Age_yhat_minerr_cutoff)
GEN_ROC = roc.curve(GEN_test$Voted_for_or_Not, GEN_yhat_minerr_cutoff)

roc.curve(PICERGA_test$Voted_for_or_Not, PICERGA_yhat_minerr_cutoff)
roc.curve(INC_test$Voted_for_or_Not, INC_yhat_minerr_cutoff)
roc.curve(CHUR_test$Voted_for_or_Not, CHUR_yhat_minerr_cutoff)
roc.curve(ECON_test$Voted_for_or_Not, ECON_yhat_minerr_cutoff)
roc.curve(RACE_test$Voted_for_or_Not, RACE_yhat_minerr_cutoff)
roc.curve(Age_test$Voted_for_or_Not, Age_yhat_minerr_cutoff)
roc.curve(GEN_test$Voted_for_or_Not, GEN_yhat_minerr_cutoff)

all_RUC = matrix(1:21, nrow = 7, ncol = 3)
rownames(all_RUC) = c('PICERGA', 'Income', 'Church', 'Economy', 'Race', 'Age', 'Gender')
colnames(all_RUC) = c('Area Under the curve (the large the better)','Error','Cutoff')

all_RUC["PICERGA",] = c(PICERGA_ROC$auc,PICERGA_err_minerr,cutoff)
all_RUC["Income",] = c(INC_ROC$auc,PICERGA_err_minerr,cutoff)
all_RUC["Church",] = c(PICERGA_ROC$auc,INC_err_minerr,cutoff)
all_RUC["Economy",] = c(ECON_ROC$auc,ECON_err_minerr,cutoff)
all_RUC["Race",] = c(RACE_ROC$auc,RACE_err_minerr,cutoff)
all_RUC["Age",] = c(Age_ROC$auc,Age_err_minerr,cutoff)
all_RUC["Gender",] = c(GEN_ROC$auc,GEN_err_minerr,cutoff)

sorted_all_RUC = all_RUC[order(all_RUC[,1], decreasing = T),] %>% round(rounding)
View(sorted_all_RUC)
