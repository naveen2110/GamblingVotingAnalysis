## KNN for classification 
#  this file includes that running KNN over different factors to classify 
#  whether a group of people or country will vote for Gambling or not
#  and compare which factors are the best one to use.
# Content as follow:
# Preparation
# 1. Income predictors
# 2. population, dancity, and size of countries predictors
# 3. Race predictors
# 4. Churche predictors
# 5. Economic predictors
# 6. Over all predictors
# Error Comparison of predictor groups
#############################################################################################
## Preparation  
rm(list=ls()); gc(); dev.off(dev.list()["RStudioGD"])

setwd()
dat = read.csv('Dataset_KNN.csv', stringsAsFactors=T, head=T)
library('rpart')
require('class')
library('dplyr')
library('ROSE')

#This line just changes one of the column names to align codes from different authors
names(dat)[28] <- "Voted_For"

# you can change this part 
################################
# Accuracy predictor          ##
AF = 0.5                      ##
# Kgrid                       ##
grid = seq(1, 100, 2)         ##
# Ballot_Type                 ##
Ballot_Type_selecton = 1      ##
# cutoff                      ##
cutoff = 0.6                  ##
# round final tables          ##
rounding = 4                  ##
################################

# data seperation
dat.ballot = subset(dat, Ballot_Type == Ballot_Type_selecton)

#Add binary Voted_For column
dat.ballot$Voted_For = dat.ballot$Votes_For > dat.ballot$Votes_Against

#Normalize the data
dat.ballot[,3:5] = scale(dat.ballot[,3:5])
dat.ballot[,7:25] = scale(dat.ballot[,7:25])
dat.ballot[,27] = scale(dat.ballot[,27])

# partation index
set.seed(1)
n.train = floor(nrow(dat.ballot)*0.70)
ind.train = sample(1:nrow(dat.ballot), n.train)
ind.test = setdiff(1:nrow(dat.ballot), ind.train)

# function to calculate RMSE ##
myRMSE = function(yhat.vec, ytest.vec) {
  er = yhat.vec - ytest.vec
  rmse = sqrt( mean(er^2) )
  return(rmse)
}

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

#############################################################################################
## 1. Income

Inc_dat = dat.ballot[,c('Voted_For', 'Per_Capita_Income',	'Medium_Family_Income')]


# partation data
Inc_Xtrain = Inc_dat[ind.train,2:3]
Inc_ytrain = Inc_dat[ind.train,1]
Inc_Xtest = Inc_dat[ind.test,2:3]
Inc_ytest = Inc_dat[ind.test,1]

# knn over income level
Inc_best_knn = knn.bestK(Inc_Xtrain, Inc_Xtest, Inc_ytrain, Inc_ytest, k.grid = grid, AF)
Inc_best_knn

# rerun with the best k
Inc_ypred = knn(Inc_Xtrain, Inc_Xtest, Inc_ytrain, k=Inc_best_knn$k.optimal, prob=T)
table(Inc_ytest, Inc_ypred)
Inc_err = mean(Inc_ypred != Inc_ytest)
Inc_ROC = roc.curve(Inc_ytest, Inc_ypred)


#############################################################################################
## 2. population, density, and size of countries

PDS_dat = dat.ballot[,c('Voted_For', 'Population',	'Population_Density', 'Size_of_County')]

# partation data
PDS_Xtrain = PDS_dat[ind.train,2:4]
PDS_ytrain = PDS_dat[ind.train,1]
PDS_Xtest = PDS_dat[ind.test,2:4]
PDS_ytest = PDS_dat[ind.test,1]

# knn over countries
PDS_best_knn = knn.bestK(PDS_Xtrain, PDS_Xtest, PDS_ytrain, PDS_ytest, k.grid = grid, AF)
PDS_best_knn

# rerun with the best k
PDS_ypred = knn(PDS_Xtrain, PDS_Xtest, PDS_ytrain, k=PDS_best_knn$k.optimal, prob=T)
table(PDS_ytest, PDS_ypred)
PDS_err = mean(PDS_ypred != PDS_ytest)
PDS_ROC = roc.curve(PDS_ytest, PDS_ypred)

#############################################################################################
## 3. RACE

RACE_dat = dat.ballot[,c('Voted_For', 'Percent_White',	'Percent_Black', 'Percent_Other')]

# partation data
RACE_Xtrain = RACE_dat[ind.train,2:4]
RACE_ytrain = RACE_dat[ind.train,1]
RACE_Xtest = RACE_dat[ind.test,2:4]
RACE_ytest = RACE_dat[ind.test,1]

# knn over RACE
RACE_best_knn = knn.bestK(RACE_Xtrain, RACE_Xtest, RACE_ytrain, RACE_ytest, k.grid = grid, AF)
RACE_best_knn

# rerun with the best k
RACE_ypred = knn(RACE_Xtrain, RACE_Xtest, RACE_ytrain, k=RACE_best_knn$k.optimal, prob=T)
table(RACE_ytest, RACE_ypred)
RACE_err = mean(RACE_ypred != RACE_ytest)
RACE_ROC = roc.curve(RACE_ytest, RACE_ypred)

#############################################################################################
## 4. Churches

Chur_dat = dat.ballot[,c('Voted_For', 'No_of_Churches',	'No_of_Church_Members')]

# partation data
Chur_Xtrain = Chur_dat[ind.train,2:3]
Chur_ytrain = Chur_dat[ind.train,1]
Chur_Xtest = Chur_dat[ind.test,2:3]
Chur_ytest = Chur_dat[ind.test,1]

# knn over Churches
Chur_best_knn = knn.bestK(Chur_Xtrain, Chur_Xtest, Chur_ytrain, Chur_ytest, k.grid = grid, AF)
Chur_best_knn

# rerun with the best k
Chur_ypred = knn(Chur_Xtrain, Chur_Xtest, Chur_ytrain, k=Chur_best_knn$k.optimal, prob=T)
table(Chur_ytest, Chur_ypred)
Chur_err = mean(Chur_ypred != Chur_ytest)
Chur_ROC = roc.curve(Chur_ytest, Chur_ypred)

#############################################################################################
## 5. Economic predictors

Econ_dat = dat.ballot[,c('Voted_For', 'Poverty_Level_Rate',	'Unemployment_Rate')]

# partation data
Econ_Xtrain = Econ_dat[ind.train,2:3]
Econ_ytrain = Econ_dat[ind.train,1]
Econ_Xtest = Econ_dat[ind.test,2:3]
Econ_ytest = Econ_dat[ind.test,1]

# knn over Economic predictors
Econ_best_knn = knn.bestK(Econ_Xtrain, Econ_Xtest, Econ_ytrain, Econ_ytest, k.grid = grid, AF)
Econ_best_knn

# rerun with the best k
Econ_ypred = knn(Econ_Xtrain, Econ_Xtest, Econ_ytrain, k=Econ_best_knn$k.optimal, prob=T)
table(Econ_ytest, Econ_ypred)
Econ_err = mean(Econ_ypred != Econ_ytest)
Econ_ROC = roc.curve(Econ_ytest, Econ_ypred)

#############################################################################################
## 6. Age

Age_dat = dat.ballot[,c('Voted_For', 'Age_Less_than_18', 'Age_18_24', 'Age_24_44', 'Age_44_64', 'Age_Older_than_64')]

# partation data
Age_Xtrain = Age_dat[ind.train,2:6]
Age_ytrain = Age_dat[ind.train,1]
Age_Xtest = Age_dat[ind.test,2:6]
Age_ytest = Age_dat[ind.test,1]

# knn over Age
Age_best_knn = knn.bestK(Age_Xtrain, Age_Xtest, Age_ytrain, Age_ytest, k.grid = grid, AF)
Age_best_knn

# rerun with the best k
Age_ypred = knn(Age_Xtrain, Age_Xtest, Age_ytrain, k=Age_best_knn$k.optimal, prob=T)
table(Age_ytest, Age_ypred)
Age_err = mean(Age_ypred != Age_ytest)
Age_ROC = roc.curve(Age_ytest, Age_ypred)

#############################################################################################
## 6. Over all predictors

Overall_dat = dat.ballot[,c('Voted_For', 
                        'Per_Capita_Income',	'Medium_Family_Income',
                        'Population',	'Population_Density', 'Size_of_County',
                        'Percent_White',	'Percent_Black', 'Percent_Other',
                        'No_of_Churches',	'No_of_Church_Members', 
                        'Poverty_Level_Rate',	'Unemployment_Rate', 
                        'Age_Less_than_18', 'Age_18_24', 'Age_24_44', 'Age_44_64', 'Age_Older_than_64')]

# partation data
Overall_Xtrain = Overall_dat[ind.train,2:18]
Overall_ytrain = Overall_dat[ind.train,1]
Overall_Xtest = Overall_dat[ind.test,2:18]
Overall_ytest = Overall_dat[ind.test,1]

# knn over Age
Overall_best_knn = knn.bestK(Overall_Xtrain, Overall_Xtest, Overall_ytrain, Overall_ytest, k.grid = grid, AF)
Overall_best_knn

# rerun with the best k
Overall_ypred = knn(Overall_Xtrain, Overall_Xtest, Overall_ytrain, k=Overall_best_knn$k.optimal, prob=T)
table(Overall_ytest, Overall_ypred)
Overall_err = mean(Overall_ypred != Overall_ytest)
Overall_ROC = roc.curve(Overall_ytest, Overall_ypred)


#############################################################################################
# Error Comparison of predictor groups

all_RUC = matrix(1:21, nrow = 7, ncol = 3)
rownames(all_RUC) = c('Income', 'Papulation', 'Race', 'Church', 'Economy', 'Age', 'Overall')
colnames(all_RUC) = c('Area Under the curve (the large the better)','Error','Cutoff')

all_RUC["Income",] = c(Inc_ROC$auc,Inc_err,cutoff)
all_RUC["Papulation",] = c(PDS_ROC$auc,PDS_err,cutoff)
all_RUC["Race",] = c(RACE_ROC$auc,RACE_err,cutoff)
all_RUC["Church",] = c(Chur_ROC$auc,Chur_err,cutoff)
all_RUC["Economy",] = c(Econ_ROC$auc,Econ_err,cutoff)
all_RUC["Age",] = c(Age_ROC$auc,Age_err,cutoff)
all_RUC["Overall",] = c(Overall_ROC$auc,Overall_err,cutoff)

sorted_all_RUC_KNN = all_RUC[order(all_RUC[,1], decreasing = T),] %>% round(rounding)

View(sorted_all_RUC_KNN)

#############################################################################################
# now we will try to improve the error rates of the overall KNN
# We will manipulate some columns and the minimum decision percent
#re-use dat.ballot from earlier and remove the ballot type from the columns
dat3=subset(dat.ballot, select= -c(Ballot_Type))

# We will split the training and validation by a 70-30 split
set.seed(21)
n.train_1 = floor( nrow(dat3)*0.70 )

ind.train_1 = sample(1:nrow(dat3), n.train_1)
ind.test_1 = setdiff(1:nrow(dat3), ind.train_1)

#Then, I remove the voted_For column and make it my results vectors
results_1=subset(dat3, select= c(Voted_For))
dat3=subset(dat3, select= -c(Voted_For))

########################################################## 
# CHANGEABLE SECTION
# In this section I test removing other columns from dat3 to try and improve the errors and TPR and TNR
# I also set the cutoff value for trying to find the best KNN
dat3_1=dat3

dat3_1=subset(dat3, select= -c(No_of_Churches,	No_of_Church_Members))

CV=.2

##########################################################
#Now I will separate the data into training and testing data
Xtrain_1 = dat3_1[ind.train_1,]
Xtest_1 = dat3_1[ind.test_1,]
ytrain_1 = results_1[ind.train_1,]
ytest_1 = results_1[ind.test_1,]

#Now we find the best ks for the chosen ballot type

results_3_1 = knn.bestK(Xtrain_1, Xtest_1, ytrain_1, ytest_1, seq(1, 157, 2), CV)

#After finding the best ks, we save the data and output it

bestk_1=results_3_1$k.optimal

## run with the best k
ypred_1 = knn(Xtrain_1, Xtest_1, ytrain_1, k=bestk_1, prob=T)
matrix_1=table(ytest_1, ypred_1)

error_1=((matrix_1[2]+matrix_1[3])/(matrix_1[1]+matrix_1[2]+matrix_1[3]+matrix_1[4]))
  
#True positive rate / Sensitivity for ballot 1
TPR_1= matrix_1[1]/(matrix_1[1]+matrix_1[3])

# True Negative Rate / Specificity for ballot 1
TNR_1= matrix_1[4]/(matrix_1[2]+matrix_1[4])


#Results comparison
print(paste("The best k for ballot 1 is ",bestk_1))
print(paste("The minimum error for ballot 1 is ",error_1))
print(paste("The TPR for ballot 1 is ",TPR_1))
print(paste("The TNR for ballot 1 is ",TNR_1))
