#Conducting data cleaning and Preprocessing to analyze voter behaviour in 18 states 
#with respect to their Votes for and against Ballot Types :1 - Gambling , 2- Wagering

#DATA CLEANING STARTS HERE
rm(list=ls()); gc(); dev.off(dev.list()["RStudioGD"])

setwd()
dat <- read.csv('Gaming Data Set.csv', head=T, stringsAsFactors=F, na.strings='')

#check data types of all variables are modify as required
summary(dat) 
#Convert character data types to numeric data types. The as.numeric() function replaces all
#the character values in the column to NA. this can be used to detect cells with 'NULL' value.
#The is.na() function can then be used to delete rows containing NA values.
dat<-transform(dat,Per_Capita_Income  = as.numeric(gsub('\\$|,', '', Per_Capita_Income)),
               Medium_Family_Income = as.numeric(gsub('\\$|,', '', Medium_Family_Income)),
               No_of_Church_Members=as.numeric(No_of_Church_Members))
matrix.na <- is.na(dat) #create a matrix which shows TRUE if there's a missing value

dat1 <- na.omit(dat) # remove rows with empty/'NA' values.

#Check if there are large discrepancies between Population and the sum of age demographics
#Need line of code to add a column called "delta_population" to measure discrepancies
delta_pop=dat1$Population - (dat1$Age_Less_than_18+dat1$Age_18_24+dat1$Age_24_44+dat1$Age_44_64+dat1$Age_Older_than_64)

#Now to scale the discrepancy against population 
delta_percent=delta_pop/dat1$Population

#Add delta_percent as a column 
dat1=cbind(dat1,delta_percent)

#Write a function to replace outlier or unwanted values with 'NA'
#install.packages('ggplot2')
library(ggplot2)
#install.packages('data.table')
library(data.table)

outlierReplace = function(dataframe, cols, rows, newValue = NA) {
  if (any(rows)) {
    set(dataframe, rows, cols, newValue)
  }
}

#Mark rows that have greater than 10% discrepancies 
outlierReplace(dat1,'delta_percent',which(dat1$delta_percent >= 0.1), newValue = NA)
outlierReplace(dat1,'delta_percent',which(dat1$delta_percent<= -0.1), newValue = NA)

#remove rows 
dat2<-na.omit(dat1)

#Remove the column delta_percent that we added for this cleaning step
dat2[28] <- NULL

#adjust remaining minor discrepancies in data
dat2$Total_Votes<-dat2$Votes_For+dat2$Votes_Against
dat2$Population<- dat2$Age_Less_than_18+dat2$Age_18_24+dat2$Age_24_44+dat2$Age_44_64+
                  dat2$Age_Older_than_64
dat2$Population_Density<- dat2$Population / dat2$Size_of_County

#Scale age demographics to population
dat2[,21:25]=dat2[,21:25]/dat2[,7] 

#Scale No_of_Churches and No_of_Church_Members to population
dat2[,17:18]=dat2[,17:18]/dat2[,7]

summary(dat2) 

# Visually check outliers of scaled variables 
hist(dat2$Per_Capita_Income)
hist(dat2$Medium_Family_Income)
hist(dat2$Population_Density)
hist(dat2$No_of_Churches)
hist(dat2$No_of_Church_Members)

#After checking for outliers, all variables look good except for No_of_Church_Members  
#there appear to be several rows where church members:population is >1, 

#Mark rows where there are more church members than population
outlierReplace(dat2,'No_of_Church_Members',which(dat2$No_of_Church_Members >1), newValue = NA)

dat3=na.omit(dat2)

summary(dat3)

#VOTER TURNOUT: Because population includes people aged 0-17, 
#rows where total votes > 70% of the population are to be marked and removed
outlierReplace(dat3,'Total_Votes',which(dat3$Total_Votes >= 0.7*dat3$Population), newValue = NA)

#Remove marked outliers
dat4=na.omit(dat3) 

#Final cleaned file with 1251 observations
# write.csv(dat4, file = "Dataset_MLR.csv", row.names = FALSE)

##DATA CLEANING ENDS HERE

########################################################################################################
########################################################################################################

#MLR BEGINS HERE
#Adding a new row that is: (votes_for - votes_against)/population
#purpose: obtain a continuous dependent variable measuring the degree counties vote yes, scaled to population
Delta_Votes = dat4$Votes_For - dat4$Votes_Against

Deltavotes_Population = Delta_Votes/dat4$Population

#Add Deltavotes_Population as a column 
dat5 = cbind(dat4,Deltavotes_Population)

#Check new column
summary(dat5)
#Looks good, min is 29% more of population voted against, max is 20% more of pop voted for

#Remove columns that ar not variables and all unscaled data
dat5[1:5] <- NULL

#Select Ballot 1
Ballot_Type_selecton = 1 
dat5ballot1 = subset(dat5, Ballot_Type == Ballot_Type_selecton)

#Partition data 
set.seed(1)
n.train = floor( nrow(dat5ballot1)*.6 )
id.train = sample(1:nrow(dat5ballot1), n.train)
id.test = setdiff(1:nrow(dat5ballot1), id.train)

#Run MLR with Deltavotes_Population as dependent variable
obj.null = lm(Deltavotes_Population ~ 1, dat = dat5ballot1[id.train, ])
obj.full = lm(Deltavotes_Population ~ ., dat = dat5ballot1[id.train, ])

#Forward selection
MLRforward = step(obj.null, scope=list(lower=obj.null, upper=obj.full), direction='forward')

summary(MLRforward)

#MLR both 
MLRboth = step(obj.null, scope=list(lower=obj.null, upper=obj.full), direction='both')

summary(MLRboth)

#MLR backward
MLRback = step(obj.full, scope=list(lower=obj.null, upper=obj.full), direction='backward')

summary(MLRback)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Check validation data to find out which selection method produces the best error rate

rmse = function(yhat.vec, ytest.vec) {
  er = yhat.vec - ytest.vec
  rmse = sqrt( mean(er^2) )
  return(rmse)
}

# forward
yhat1 = predict(MLRforward, newdata = dat5ballot1[id.test, ])
rmse(dat5ballot1[id.test, 'Deltavotes_Population'], yhat1) ## RMSE for test data

# stepwise
yhat3 = predict(MLRboth, newdata = dat5ballot1[id.test, ])
rmse(dat5ballot1[id.test, 'Deltavotes_Population'], yhat3) ## RMSE for test data

# backward
yhat2 = predict(MLRback, newdata = dat5ballot1[id.test, ])
rmse(dat5ballot1[id.test, 'Deltavotes_Population'], yhat2) ## RMSE for test data


############## FINISHED WITH BALLOT 1 ##############################################################
################## BEGIN BALLOT 2 ##################################################################

#Select Ballot 2
Ballot_Type_selecton = 2 
dat5ballot2 = subset(dat5, Ballot_Type == Ballot_Type_selecton)

#Partition data 
set.seed(1)
n.train = floor( nrow(dat5ballot2)*.6 )
id.train = sample(1:nrow(dat5ballot1), n.train)
id.test = setdiff(1:nrow(dat5ballot1), id.train)

#Run MLR with Deltavotes_Population as dependent variable
obj.null = lm(Deltavotes_Population ~ 1, dat = dat5ballot2[id.train, ])
obj.full = lm(Deltavotes_Population ~ ., dat = dat5ballot2[id.train, ])

#Forward selection
MLRforward = step(obj.null, scope=list(lower=obj.null, upper=obj.full), direction='forward')

summary(MLRforward)

#MLR both 
MLRboth = step(obj.null, scope=list(lower=obj.null, upper=obj.full), direction='both')

summary(MLRboth)

#MLR backward
MLRback = step(obj.full, scope=list(lower=obj.null, upper=obj.full), direction='backward')

summary(MLRback)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Check validation data to find out which selection method produces the best error rate

# forward
yhat1 = predict(MLRforward, newdata = dat5ballot2[id.test, ])
rmse(dat5ballot2[id.test, 'Deltavotes_Population'], yhat1) ## RMSE for test data

# stepwise
yhat3 = predict(MLRboth, newdata = dat5ballot2[id.test, ])
rmse(dat5ballot2[id.test, 'Deltavotes_Population'], yhat3) ## RMSE for test data

# backward
yhat2 = predict(MLRback, newdata = dat5ballot2[id.test, ])
rmse(dat5ballot2[id.test, 'Deltavotes_Population'], yhat2) ## RMSE for test data

