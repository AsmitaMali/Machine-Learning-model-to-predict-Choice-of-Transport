getwd()
car = read.csv('Cars.csv',header = T)
View(car)

#EDA
dim(car)
#44 records and 9 variables

summary(car)
View(car)

str(car)
## Factor conversion - Engineer,MBA and license 
car$Engineer = as.factor(car$Engineer)
car$MBA = as.factor(car$MBA)
car$license = as.factor(car$license)

names(car)

#data structure post conversion
str(car)


# summary of the dataset
summary(car)
#There is one missing data for the column "MBA"
#this is creating problem in fittend values
#need to impute this


library(DMwR)
#there is one NA value in the table; activity to remove NA knn imputation
#?knnImputation
car = knnImputation(car)

summary(car)

library(funModeling)
plot_num(car)
#age range from 20 to 40, most in the range of 25 to 30
#Work experience from 0 to 20+ most in the range of 4 to 9 years
#distance normally distribute range from 5-20. Most people in 7-15 away 


# outlier check 

#age
boxplot(car$Age, main="Age")

#work.exp
boxplot(car$Work.Exp, main="Work Experience")

#salary
boxplot(car$Salary, main="Salary")

#distance
boxplot(car$Distance, main="Distance")

#There are outliers in most of the numeric cells.
#Will not be treating outliers as of now

#correlation plot
library(GGally)
ggcorr(car)
# Age Correlated with Work Exp and Salary
# Work Experience with Salary
#indicates multicollinearity



#as we have to predict if the employee uses car or not we plan to create a variable
#Car use of 2 levels 0 = not using car & 1 = using car
car$Target = ifelse(car$Transport =='Car',1,0)
car$Target<-as.factor(car$Target)



#check the how each variable impacts Target

#age
boxplot(car$Age~car$Target)
t.test(car$Age[car$Target==1],car$Age[car$Target==0])
#impact


#salary
boxplot(car$Salary~car$Target)
t.test(car$Salary[car$Target==1],car$Salary[car$Target==0])
#impact


#work experience 
boxplot(car$Work.Exp~car$Target)
t.test(car$Work.Exp[car$Target==1],car$Work.Exp[car$Target==0])
#impact


#distance
boxplot(car$Distance~car$Target)
t.test(car$Distance[car$Target==1],car$Distance[car$Target==0])
#impact




#Gender - ChiSquare
chisq.test(table(car$Gender,car$Target))

#Engineer - ChiSquare
chisq.test(table(car$Engineer,car$Target))

#MBA - ChiSquare
chisq.test(table(car$MBA,car$Target))

#license - ChiSquare
chisq.test(table(car$license,car$Target))



# training and Testing splitting data set
set.seed(123)
totalrows <- nrow(car)
training.sample = round(((totalrows)*70/100),0)
s <- sample(totalrows, size = training.sample)
car_train <- car[s,]
car_test <- car[-s,]

#to see if the split has happened properly
nrow(car_train)
nrow(car_test)


#Response Rate
nrow(car_train[car_train$Target=="1",])/nrow(car_train)
nrow(car_test[car_test$Target=="1",])/nrow(car_test)


# logistic regression model
car_glm_model1 = glm(Target ~ Age+Gender+Salary+Engineer+MBA+Work.Exp+Distance+license, data=car_train, family = binomial)
summary(car_glm_model1)

#Engg is not shown as significant

#Multicollinearity 
library(car)
vif(car_glm_model1)
#work experince show high VIF remove it


#model removing Engineer
car_glm_model2 = glm(Target ~ Age+Gender+MBA+Salary+Distance+license, data=car_train, family = binomial)
summary(car_glm_model2)
#all signifincant

#Multicollinearity 
vif(car_glm_model2)
#nothing above 10 retain all variables


# assigning  probablities and class
car_train$fittedvalue <-car_glm_model2$fitted.values
car_train$prediction  <-ifelse(car_train$fittedvalue >0.5, 1, 0)


#confusion matrix
library(caret)
library(e1071)
confusionMatrix( as.factor(car_train$prediction),as.factor(car_train$Target))

# Other Model Performance Measures 
library(ROCR)
library(ineq)
pred <- ROCR::prediction(car_train$fittedvalue, car_train$Target)
perf <- performance(pred, "tpr", "fpr")
KS <- max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
auc <- performance(pred,"auc"); 
auc <- as.numeric(auc@y.values)
gini = ineq(car_train$fittedvalue, type="Gini")

with( car_train, table(Target, as.factor(prediction)  ))
auc
KS
gini = 2 * auc - 1
gini


#testing
car_test$fittedvalue <-predict(car_glm_model2, newdata=car_test, type = "response")

## Assgining 0 / 1 class based on certain threshold
car_test$prediction <- ifelse(car_test$fittedvalue >0.5, 1, 0)


#Confusion Matrix testing
confusionMatrix( as.factor(car_test$prediction),as.factor(car_test$Target))

# Other Model Performance Measures  - testing
pred <- ROCR::prediction(car_test$fittedvalue, car_test$Target)
perf <- performance(pred, "tpr", "fpr")
KS <- max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
auc <- performance(pred,"auc"); 
auc <- as.numeric(auc@y.values)
gini = ineq(car_test$fittedvalue, type="Gini")

with( car_test, table(Target, as.factor(prediction)  ))
auc
KS
gini = 2 * auc - 1
gini


##KNN

# Feature Scaling
car_trainknn = car[s,]
car_testknn = car[-s,]
car_trainknn[-c(2,3,4,8,9,10)] = scale(car_trainknn[-c(2,3,4,8,9,10)])
car_testknn[-c(2,3,4,8,9,10)] = scale(car_testknn[-c(2,3,4,8,9,10)])
summary(car_trainknn$Gender)

# Encoding the Gender feature as factor with level 0 = Male, 1 = Female
car_trainknn$Gender = ifelse(car_trainknn$Gender == 'Female',1,0)
car_trainknn$Gender <- as.factor(car_trainknn$Gender)
  
#Removing the Transport variable as it is no longer required
car_trainknn <- car_trainknn[,-9]
car_testknn <- car_testknn[,-9]

#Converting all Factor & Ineteger variables to Numeric
car_trainknn$Age = as.numeric(car_trainknn$Age)
car_trainknn$Gender = as.numeric(car_trainknn$Gender)
car_trainknn$Engineer = as.numeric(car_trainknn$Engineer)
car_trainknn$MBA = as.numeric(car_trainknn$MBA)
car_trainknn$license = as.numeric(car_trainknn$license)
car_trainknn$Target = as.numeric(car_trainknn$Target)


car_testknn$Age = as.numeric(car_testknn$Age)
car_testknn$Gender = as.numeric(car_testknn$Gender)
car_testknn$Engineer = as.numeric(car_testknn$Engineer)
car_testknn$MBA = as.numeric(car_testknn$MBA)
car_testknn$license = as.numeric(car_testknn$license)
car_testknn$Target = as.numeric(car_testknn$Target)

# Fitting K-NN to the Training set and Predicting the Test set results
library(class)
?knn
car_knnpred = knn(train = car_trainknn[, -9],
             test = car_testknn[, -9],
             cl = car_trainknn[, 9],
             k = 3,
             prob = F)
car_knnpred
# Making the Confusion Matrix
cmknn = table(car_testknn[, 9], car_knnpred)
cmknn

summary(car_knnpred)

#Confusion Matrix testing
confusionMatrix( as.factor(car_knnpred),as.factor(car_testknn$Target))


##Naive Bayes

# Training and testing dataset for NB
car_trainnb = car[s,]
car_testnb = car[-s,]

#Removing the Transport variable as it is no longer required
car_trainnb <- car_trainnb[,-9]
car_testnb <- car_testnb[,-9]


# Fitting NB to the Training set
library(e1071)
?naiveBayes
classifier = naiveBayes(x = car_trainnb[-9],
                        y = car_trainnb$Target)

# Predicting the Test set results
car_prednb = predict(classifier, newdata = car_testnb[-9])

# Making the Confusion Matrix
cmnb = table(car_testnb[, 9], car_prednb)
cmnb

#Confusion Matrix testing
confusionMatrix( as.factor(car_knnpred),as.factor(car_testknn$Target))

#Check for multicollinearity
library(GGally)
ggcorr(car)
# Age Correlated with Work Exp and Salary
# Work Experience with Salary
#indicates multicollinearity

#NB assumes independence of predictor variables, but multicollinearity is observed here
#hence, nb is not applicable, as it will over inflate their effect

#Applying NB on dataset without correlated variables - Work Exp., Salary
classifier2 = naiveBayes(x = car_trainnb[-c(5,6,9)],
                        y = car_trainnb$Target)

# Predicting the Test set results
car_prednb2 = predict(classifier2, newdata = car_testnb[-c(5,6,9)])

summary(car_prednb2)
# Making the Confusion Matrix
cmnb2 = table(car_testnb[, 9], car_prednb2)
cmnb2

#Confusion Matrix testing
confusionMatrix( as.factor(car_prednb2),as.factor(car_testnb[, 9]))


##Boosting

# Creating test and train dataset 
car_trainboost = car[s,]
car_testboost = car[-s,]
car_trainboost$Gender = ifelse(car_trainboost$Gender == 'Female',1,0)
car_testboost$Gender = ifelse(car_testboost$Gender == 'Female',1,0)

#Removing the Transport variable as it is no longer required
car_trainboost <- car_trainboost[,-9]
car_testboost <- car_testboost[,-9]

# Fitting XGBoost to the Training set
# install.packages('xgboost')
library(xgboost)

#Training the XgBoost model on train dataset
model <- train(
  Target ~., data = car_trainboost, method = "xgbTree",
  trControl = trainControl("cv", number = 10)
)

#Predicting the target variable
car_predboost = predict(model, newdata =car_testboost)
predicted.classes

#Confusion Matrix
confusionMatrix( car_predboost,car_testboost[,9])

varImp(model)

#Bagging
car_trainbagg = car[s,]
car_testbagg = car[-s,]

#Removing the Transport variable as it is no longer required
car_trainbagg <- car_trainbagg[,-9]
car_testbagg <- car_testbagg[,-9]

library(ipred)
library(rpart)
set.seed(123)
?bagging
classifier_bagg <- bagging(formula = Target ~ .,data = car_trainbagg,
  nbagg = 10,  
  coob = TRUE,
  control = rpart.control(minsplit = 2, cp = 0)
)

#Predicting
car_predbagg <- predict(classifier_bagg, newdata = car_testbagg[,-9])
car_predbagg
car_predbagg = ifelse(car_predbagg > 1,2,1)

summary(as.factor(car_predbagg))
# Making the Confusion Matrix
cmbagg = table(car_testbagg[, 9], car_predbagg)
cmbagg


#Confusion Matrix testing
confusionMatrix( as.factor(car_predbagg),as.factor(car_testbagg[, 9]))

#Importance of variables in boosting
varImp(model)