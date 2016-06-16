#_________________________________________________
#
# FRA Group assignment: Default modeling in R
#
# You are provided data on loan defaults. 
# Your objective is to build default prediction models using R.
# You would be evaluated on the following tasks:
#
#  -	Cleaning of data
#
#  -	Handling unbalanced data
#
#  -	Building model(s) on training data 
#       o	How many models? More the merrier 
#
#  -	Evaluating model performance using test data 
#
#
# Group Assignment Members:
#
# Arun Sundar
# Arun Venkatesan
# Rajagopalan Kannan
# Vijay Somanath
#
#_________________________________________________




#_________________________________________________
#
#  Clean up
#
#_________________________________________________
rm(list=ls())
gc()

#_________________________________________________
#
# Load packages
#
#_________________________________________________
require(caret)
#require(plyr)
#require(Hmisc)
require(pROC)
require(DMwR)
#require(missForest)
#require(mi)
#require(mice)
#require(car)
#require(xgboost)
require(Amelia)
require(ROCR)
require(unbalanced)
library(plyr)

#_________________________________________________
#
# Read files
#
#_________________________________________________
#setwd('c:/r/data/csv/')
train <- read.csv('fra_train.csv', header=TRUE)
test <- read.csv('fra_test.csv', header=TRUE)


#_________________________________________________
#
# Study the data set
#
#_________________________________________________
dim(train)
dim(test)
str(train)
str(test)

#_________________________________________________
#
# Check missing values
#
#_________________________________________________

# visuals
missmap(train, col=c('red', 'white'), legend=FALSE, y.cex=0.3,
        main='training data set - missingness map')
missmap(test, col=c('red', 'white'), legend=FALSE, y.cex=0.3,
        main='test data set - missingness map')

# number of missing values by attributes
sapply(train, function(x) sum(is.na(x)))
sapply(test, function(x) sum(is.na(x)))

# train we can see that ~94% of the classes are of '0' and 6% are '1'
(a <- table(train$SeriousDlqin2yrs))
prop.table(table(train$SeriousDlqin2yrs))

(b <- table(test$SeriousDlqin2yrs))
prop.table(table(test$SeriousDlqin2yrs))

pie(table(train$SeriousDlqin2yrs))


#_________________________________________________
#
# Let's do some exploratory
#
#_________________________________________________
# source('helper.R')
# hgraph(train[,-1])
# hgraph(test[,-1])

# Casenum                 Identifier
# SeriousDlqin2yrs        Categorical - outcome variable
# RevolvingUtilizationOfUnsecuredLines  Continuous (Skewed)
# DebtRatio               Continuous (Skewed)
# NumberOfOpenCreditLinesAndLoans Continuous (Skewed)
# NumberOfDependents      Categorical

ggplot(train, aes(NumberOfDependents,fill=factor(SeriousDlqin2yrs))) +
  geom_bar(position="dodge")+coord_flip()

#_________________________________________________
#
# Check for unbalanced classes
#
#_________________________________________________



# Debt ratio & Revolving utilization of unsecured loans

ggplot(train,aes(x=factor(train$SeriousDlqin2yrs),y=DebtRatio))+geom_boxplot()

ggplot(train, aes(x=factor(train$SeriousDlqin2yrs),y=train$RevolvingUtilizationOfUnsecuredLines))+geom_boxplot()


#_________________________________________________
#
# Data Pre-Processing
#
#_________________________________________________

# predictors must be numeric for roc function
train$SeriousDlqin2yrs <- factor(train$SeriousDlqin2yrs)
test$SeriousDlqin2yrs <- factor(test$SeriousDlqin2yrs)

#train$NumberOfDependents <- factor(train$NumberOfDependents)
#test$NumberOfDependents <- factor(test$NumberOfDependents)



#_________________________________________________
#
# Mode Imputation
#
#_________________________________________________


getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}


train$NumberOfDependents[is.na(train$NumberOfDependents)] <- getmode(train$NumberOfDependents)
test$NumberOfDependents[is.na(test$NumberOfDependents)] <- getmode(test$NumberOfDependents)

train$NumberOfDependents <- as.numeric(train$NumberOfDependents)
test$NumberOfDependents <- as.numeric(test$NumberOfDependents)

#outlier treatment-Revolving
train1=subset(train,(train$RevolvingUtilizationOfUnsecuredLines<3))
str(train1)
ggplot(train1, aes(x=factor(train1$SeriousDlqin2yrs),y=train1$RevolvingUtilizationOfUnsecuredLines))+geom_boxplot()
#outlier treatment - Debt ratio
train2=subset(train1,(train1$DebtRatio<=100))
str(train2)
p=ggplot(train2,aes(x=factor(train2$SeriousDlqin2yrs),y=train2$DebtRatio))+geom_boxplot()
(p)
#_________________________________________________
#
# Now check missing values
#
#_________________________________________________

sapply(train, function(x) sum(is.na(x)))
sapply(test, function(x) sum(is.na(x)))



#_________________________________________________
#
# Derived variables
#
#_________________________________________________




#_________________________________________________
#
# Log Transformation
#
#_________________________________________________



#_________________________________________________
#
# Bin - Transformation
#
#_________________________________________________






#_________________________________________________
#
# Create Dummy Variables
#
#_________________________________________________

# dummies <- dummyVars(~ NumberOfDependents,
#                      data=train)
# dummy_var <- as.data.frame(predict(dummies, train))
# train <- cbind(train[,c(1,2,3,4,5)],dummy_var)
# 
# 
# dummies <- dummyVars(~ NumberOfDependents,
#                      data=test)
# dummy_var <- as.data.frame(predict(dummies, test))
# test <- cbind(test[,c(1,2,3,4,5)],dummy_var)

#_________________________________________________
#
# Split Data
#
#_________________________________________________

# Remove S.No attribute
df <- train2
df <- df[,-1]
test <- test[,-1]

# Split 80/20
trainIndex <- createDataPartition(df$SeriousDlqin2yrs, p=0.8,list=FALSE,times=1)
df_train <- df[trainIndex,];df_valid <- df[-trainIndex,]

# Training scheme - 10-fold CV
ctrl <- trainControl(method='repeatedcv', number=10,repeats=3)

#_________________________________________________
#
# Declare outcome and features
#
#_________________________________________________
outcome <- 'SeriousDlqin2yrs'
features <- setdiff(names(df_train), outcome)



#_________________________________________________
#
# let's SMOTE on training data set, not on valid, test data sets
#
#_________________________________________________
require(unbalanced)

# 6% of data set is of class 1 and this is minority class.
prop.table(table(df_train$SeriousDlqin2yrs))
prop.table(table(df_valid$SeriousDlqin2yrs))

#Before SMOTE
prop.table(table(train2$SeriousDlqin2yrs))

ggplot(train, aes(NumberOfOpenCreditLinesAndLoans, 
                  NumberOfDependents)) +
  geom_jitter(aes(colour=factor(SeriousDlqin2yrs)))+
  scale_colour_brewer(palette = "Paired")


# unbalanced
input=df_train[2:5]
output=df_train$SeriousDlqin2yrs
train_ub <- ubOver(X=input,Y=output)
newdf_train=cbind(train_ub$X,train_ub$Y)  
names(newdf_train)[names(newdf_train)=="train_ub$Y"]="SeriousDlqin2yrs"


ggplot(newdf_train, aes(NumberOfOpenCreditLinesAndLoans, 
                     NumberOfDependents)) +
  geom_jitter(aes(colour=factor(SeriousDlqin2yrs)), width=10)+
  scale_colour_brewer(palette = "Paired")

df_train=newdf_train

prop.table(table(df_train$SeriousDlqin2yrs))
dim(df_train)

#_________________________________________________
#
# train the SMOTE data
#
#_________________________________________________


fit <- glm(SeriousDlqin2yrs~.,
           data=df_train,
           family = 'binomial')


#______________training sample accuracy
y_pred <- predict(fit, df_train, type='response')
y_pred <- floor(y_pred+0.5)
df_train$ypred <- y_pred

sum(with(df_train, table(ypred, SeriousDlqin2yrs)))
with(df_train, table(ypred, SeriousDlqin2yrs))

auc(df_train$SeriousDlqin2yrs, df_train$ypred)

#______________validation sample accuracy
y_pred <- predict(fit, df_valid, type='response')
y_pred <- floor(y_pred+0.5)
df_valid$ypred <- y_pred

sum(with(df_valid, table(ypred, SeriousDlqin2yrs)))
with(df_valid, table(ypred, SeriousDlqin2yrs))

auc(df_valid$SeriousDlqin2yrs, df_valid$ypred)

#______________Test sample accuracy
y_pred <- predict(fit, test, type='response')
y_pred <- floor(y_pred+0.5)
test$ypred <- y_pred

sum(with(test, table(ypred, SeriousDlqin2yrs)))
with(test, table(ypred, SeriousDlqin2yrs))

auc(test$SeriousDlqin2yrs, test$ypred)


#_________________________________________________
#
# unload packages
#
#_________________________________________________
detach('package:caret')
detach('package:plyr')
detach('package:Hmisc')
detach('package:pROC')
detach('package:DMwR')
detach('package:missForest')
detach('package:mi')
detach('package:mice')
detach('package:car')
detach('package:xgboost')
detach('package:Amelia')
detach('package:ggplot2')
detach('package:ROCR')


#_________________________________________________
#
#  Clean up
#
#_________________________________________________
rm(list=ls())
gc()

