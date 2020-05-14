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

pmiss = colMeans(matrix.na) # proportion of missing for each column
nmiss = rowMeans(matrix.na) # proportion of missing for each row
plot(pmiss)

source('..\\imageMatrix.R')
matrix.na_1 = matrix.na[1:500,]
myImagePlot(matrix.na_1)

dat1 <- na.omit(dat) # remove rows with empty/'NA' values.

#check all derived variables to see if they are calculated correctly
dat1$Total_Votes<-dat1$Votes_For+dat1$Votes_Against
dat1$Population<- dat1$Age_Less_than_18+dat1$Age_18_24+dat1$Age_24_44+dat1$Age_44_64+
                  dat1$Age_Older_than_64
dat1$Population_Density<- dat1$Population / dat1$Size_of_County
#Changing age columns to percentage in order to account for the high correlation between all age groups
dat1[,21:25]=dat1[,21:25]/dat1[,7] 

# check for outliers in each column
par(mfrow=c(5, 6))
hist(dat1$Votes_For)
boxplot(dat1$Votes_For)
hist(dat1$Votes_Against)
boxplot(dat1$Votes_Against)
hist(dat1$Total_Votes)
boxplot(dat1$Total_Votes)
hist(dat1$Population)
boxplot(dat1$Population)
hist(dat1$Per_Capita_Income)
boxplot(dat1$Per_Capita_Income)
hist(dat1$Medium_Family_Income)
boxplot(dat1$Medium_Family_Income)
hist(dat1$Size_of_County)
boxplot(dat1$Size_of_County)
hist(dat1$Population_Density)
boxplot(dat1$Population_Density)
hist(dat1$No_of_Churches)
boxplot(dat1$No_of_Churches)
hist(dat1$No_of_Church_Members)
boxplot(dat1$No_of_Church_Members)
hist(dat1$Age_Less_than_18)
boxplot(dat1$Age_Less_than_18)
hist(dat1$Age_18_24)
boxplot(dat1$Age_18_24)
hist(dat1$Age_24_44)
boxplot(dat1$Age_24_44)
hist(dat1$Age_44_64)
boxplot(dat1$Age_44_64)
hist(dat1$Age_Older_than_64)
boxplot(dat1$Age_Older_than_64)

#install.packages('ggplot2')
library(ggplot2)
#install.packages('data.table')
library(data.table)

#Write a function to replace outlier values above a certain number in each column with 'NA'
outlierReplace = function(dataframe, cols, rows, newValue = NA) {
  if (any(rows)) {
    set(dataframe, rows, cols, newValue)
  }
}

# replace outlier values in each column with NA
outlierReplace(dat1,'Votes_For',which(dat1$Votes_For >20000), newValue = NA)
outlierReplace(dat1,'Votes_Against',which(dat1$Votes_Against >20000), newValue = NA)
outlierReplace(dat1,'Total_Votes',which(dat1$Total_Votes >50000), newValue = NA)
outlierReplace(dat1,'Population',which(dat1$Population >300000), newValue = NA)
outlierReplace(dat1,'Per_Capita_Income',which(dat1$Per_Capita_Income >25000), newValue = NA)
outlierReplace(dat1,'Medium_Family_Income',which(dat1$Medium_Family_Income >45000), newValue = NA)
outlierReplace(dat1,'Size_of_County',which(dat1$Size_of_County >2500), newValue = NA)
outlierReplace(dat1,'Population_Density',which(dat1$Population_Density >400), newValue = NA)
outlierReplace(dat1,'No_of_Churches',which(dat1$No_of_Churches >200), newValue = NA)
outlierReplace(dat1,'No_of_Church_Members',which(dat1$No_of_Church_Members >75000), newValue = NA)
outlierReplace(dat1,'Age_Less_than_18',which(dat1$Age_Less_than_18 >75000), newValue = NA)
outlierReplace(dat1,'Age_18_24',which(dat1$Age_18_24 >30000), newValue = NA)
outlierReplace(dat1,'Age_24_44',which(dat1$Age_24_44 >100000), newValue = NA)
outlierReplace(dat1,'Age_44_64',which(dat1$Age_44_64 >50000), newValue = NA)
outlierReplace(dat1,'Age_Older_than_64',which(dat1$Age_Older_than_64 >30000), newValue = NA)


dat2=na.omit(dat1) # Remove empty values. We are left with 1074 rows

#Add Voted_For column before scaling
dat2$Voted_for_or_Not = dat2$Votes_For > dat2$Votes_Against


# file for all ROC
write.csv(dat2, file = "Dataset_ROC.csv", row.names = FALSE)

# File for CART

write.csv(dat2, file = "Dataset_CART.csv", row.names = FALSE)

# generalization for age columns

church_pop = sum(dat2$No_of_Church_Member, dat2$No_of_Churches)
age_pop = sum(dat2$Age_Less_than_18,dat2$Age_18_24,dat2$Age_24_44,
              dat2$Age_44_64,dat2$Age_Older_than_64)

dat2$No_of_Church_Members = dat2$No_of_Church_Members/church_pop
dat2$No_of_Churches = dat2$No_of_Churches/church_pop

dat2$Age_Less_than_18 = dat2$Age_Less_than_18 / age_pop
dat2$Age_18_24= dat2$Age_18_24 / age_pop
dat2$Age_24_44= dat2$Age_24_44 / age_pop
dat2$Age_44_64= dat2$Age_44_64 / age_pop
dat2$Age_Older_than_64= dat2$Age_Older_than_64 / age_pop

# File for CART and KNN

# Normalize the data
dat2[,3:5] = scale(dat2[,3:5])
dat2[,7:25] = scale(dat2[,7:25])
dat2[,27] = scale(dat2[,27])

# Export to csv file the new data frame with outlier rows removed and no column numbers

write.csv(dat2, file = "Dataset_LOG.csv", row.names = FALSE)

write.csv(dat2, file = "Dataset_KNN.csv", row.names = FALSE)
# DATA CLEANING ENDS HERE

