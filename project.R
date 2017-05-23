rm(list=ls())
mydata <- read.csv('C:/ISEN 613/Project/attrition.csv',header=T,sep=',')
attach(mydata)
mydata$Education <- as.factor(mydata$Education)
mydata$EnvironmentSatisfaction <- as.factor(mydata$EnvironmentSatisfaction)
mydata$JobInvolvement <- as.factor(mydata$JobInvolvement)
mydata$JobLevel <- as.factor(mydata$JobLevel)
mydata$JobSatisfaction <- as.factor(mydata$JobSatisfaction)
mydata$PerformanceRating <- as.factor(mydata$PerformanceRating)
mydata$RelationshipSatisfaction <- as.factor(mydata$RelationshipSatisfaction)
mydata$StockOptionLevel <- as.factor(mydata$StockOptionLevel)
mydata$WorkLifeBalance <- as.factor(mydata$WorkLifeBalance)

detach(mydata)

## prepare data for clustering

rem <- c("Attrition","EmployeeCount","EmployeeNumber",'Over18','StandardHours')
clust.data <- mydata[ , !(names(mydata) %in% rem)]

### check number of clusters with hierarchical and then do k means
library(dummies)
clust.data.new <- dummy.data.frame(clust.data)
y <- as.matrix(clust.data.new)
hc <- hclust(d=dist(y),method = 'average')
hc_cluster<- cutree(hc,k=2)
mydata$cluster1 <- hc_cluster

dev.new()
par(mfrow=c(1,1))
plot(hc,cex=.1)
dev.off()


### identify how the observations are clustered.

par(mfrow=c(1,2))
plot(mydata$BusinessTravel[mydata$cluster1==1])
plot(mydata$BusinessTravel[mydata$cluster1==2])


plot(mydata$Department[mydata$cluster1==1])
plot(mydata$Department[mydata$cluster1==2])


plot(mydata$EducationField[mydata$cluster1==1])
plot(mydata$EducationField[mydata$cluster1==2])


plot(mydata$Gender[mydata$cluster1==1])
plot(mydata$Gender[mydata$cluster1==2])


plot(mydata$JobInvolvement[mydata$cluster1==1])
plot(mydata$JobInvolvement[mydata$cluster1==2])

### clear distinction

plot(mydata$JobRole[mydata$cluster1==1])
plot(mydata$JobRole[mydata$cluster1==2])


plot(mydata$JobSatisfaction[mydata$cluster1==1])
plot(mydata$JobSatisfaction[mydata$cluster1==2])


### hierarchical.

par(mfrow=c(1,1))
with(mydata,boxplot(mydata$HourlyRate ~ mydata$cluster1,xlab='cluster'))
with(mydata,boxplot(mydata$MonthlyIncome ~ mydata$cluster1,xlab='cluster'))
with(mydata,boxplot(mydata$MonthlyRate ~ mydata$cluster1,xlab='cluster'))
with(mydata,boxplot(mydata$NumCompaniesWorked ~ mydata$cluster1,xlab='cluster'))
with(mydata,boxplot(mydata$PercentSalaryHike ~ mydata$cluster1,xlab='cluster'))
with(mydata,boxplot(mydata$TotalWorkingYears ~ mydata$cluster1,xlab='cluster'))
with(mydata,boxplot(mydata$YearsAtCompany ~ mydata$cluster1,xlab='cluster'))
with(mydata,boxplot(mydata$YearsInCurrentRole ~ mydata$cluster1,xlab='cluster'))

table(mydata$cluster1)
### end of clustering
## the data is divided into into 2 clusters. cluster 2 has more experienced individuals,who were 
## with the company for a long time and mostly are in senior positions like managers and directors
## cluster 1 has employees who are comparatively less experienced and new to the comapany


sum(mydata$Attrition=='Yes')

## subset selection
rem <- c("EmployeeCount","EmployeeNumber",'Over18','StandardHours','cluster1')
subset.data <- mydata[ , !(names(mydata) %in% rem)]

attach(subset.data)

library(glmnet)

 for(i in 1:31){
  i <- (class(subset.data[,i])=='factor')
  print(i)}

names(subset.data[,i])


x <- model.matrix(Attrition ~ BusinessTravel+Department+Education+EducationField+EnvironmentSatisfaction+Gender+JobInvolvement+JobLevel+JobRole+JobSatisfaction+MaritalStatus+OverTime+PerformanceRating+RelationshipSatisfaction+StockOptionLevel+WorkLifeBalance,subset.data)[,-1]
xmat <- as.matrix(data.frame(Age,DailyRate,DistanceFromHome,HourlyRate,MonthlyIncome,MonthlyRate,NumCompaniesWorked,PercentSalaryHike,TotalWorkingYears,TrainingTimesLastYear,YearsAtCompany,YearsInCurrentRole,YearsSinceLastPromotion,YearsWithCurrManager,x))

set.seed(1)
train <- sample(1:nrow(x), nrow(x)/2)
test <- (-train)
Attrition.test <- Attrition[test]
val.lambda <- 10^seq(10,-2,length=100)


set.seed(1)
lasso.mod <- glmnet(xmat[train,],Attrition[train],alpha=1,lambda = val.lambda,family = 'binomial')
plot(lasso.mod)
par(mfrow=c(1,1))
cv.lasso=cv.glmnet(x[train,],Attrition[train],alpha=1,family='binomial')
plot(cv.lasso)
bestlam=cv.lasso$lambda.min

lasso.pred=predict(lasso.mod,s=bestlam,newx=xmat[test,])


out=glmnet(xmat,Attrition,alpha=1,lambda= val.lambda,family='binomial')
lasso.coef=predict(out,type="coefficients",s=bestlam)[1:30,]

length(lasso.coef)
lasso.coef[lasso.coef!=0]
length(lasso.coef[lasso.coef!=0])
names(lasso.coef[lasso.coef!=0])

detach(subset.data)
### lasso reduced the model to 14 from 30 variables. yayy!!

attach(mydata)

inc <- c('Attrition','Age','DailyRate','DistanceFromHome','MonthlyIncome','NumCompaniesWorked','TotalWorkingYears','TrainingTimesLastYear','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager','BusinessTravel','Department','EducationField','EnvironmentSatisfaction')
data.use <- mydata[ , (names(mydata) %in% inc)]
dim(data.use)


#### boosting on reduced model

set.seed(1001)
train <- sample(nrow(data.use),1100)
data.train <- data.use[train,]
data.test <- data.use[-train,]
sum(data.train$Attrition=='Yes')
sum(data.test$Attrition=='Yes')

library(gbm)
set.seed(1001)
boost.attrition <- gbm(Attrition~.,data = data.train,distribution = 'multinomial',n.trees=5000,interaction.depth = 4,shrinkage = .005,cv.folds = 10)
pred.boost <- predict(boost.attrition,data.test,n.trees=5000)
pred1.boost <- apply(pred.boost, 1, which.max)
pred1.boost[pred1.boost==1] <- 0
pred1.boost[pred1.boost==2] <- 1
cm.boost <- table(data.test$Attrition,pred1.boost)
accuracy <- (cm.boost[1,1]+cm.boost[2,2])/(sum(cm.boost))
spec <- cm.boost[1,1]/(cm.boost[1,1]+cm.boost[1,2])
sens <- cm.boost[2,2]/(cm.boost[2,1]+cm.boost[2,2])
bacc <-  .40*sens+.60*spec

### fitting logistic regression model ## 73.37% for logistic, with threshold .25
logit.reg <- glm(data.train$Attrition ~ .,data = data.train,family = binomial)
summary(logit.reg)
logit.probs <- predict(logit.reg,data.test,type='response')
logit.pred <- rep(0,nrow(data.test))
logit.pred[logit.probs >= .5] =1
table(logit.pred,data.test$Attrition)

## BACC (balanced accuracy) curve for imbalance distribution
threshold <- seq(from = 0, to = .5, by = 0.01)
Accuracy <- rep(0,length(threshold))
bacc <- rep(0,length(threshold))
sens <- as.numeric(rep(0,length(threshold)))
spec <- as.numeric(rep(0,length(threshold)))

## BACC (balanced accuracy) curve for imbalance distribution
for(i in 1:length(threshold)){
  logit.pred <- rep(0,nrow(data.test))
  logit.pred[logit.probs >= threshold[i]] =1
  sens[i] <- mean(logit.pred[data.test$Attrition=='Yes']==1)
  spec[i] <- mean(logit.pred[data.test$Attrition=='No']==0)
  bacc[i] <- .40*sens[i]+.60*spec[i]
}

plot(threshold,bacc,type="l",ylab='balanced accuracy',main='BACC Curve')
threshold[which.max(bacc)]
bacc[which.max(bacc)]

logit.pred <- rep(0,nrow(data.test))
logit.pred[logit.probs >= .25] =1
cm.logit <- table(logit.pred,data.test$Attrition)
accuracy <- (cm.logit[1,1]+cm.logit[2,2])/(sum(cm.logit))
spec <- cm.logit[1,1]/(cm.logit[1,1]+cm.logit[1,2])
sens <- cm.logit[2,2]/(cm.logit[2,1]+cm.logit[2,2])
bacc <-  .40*sens+.60*spec

###knn

library(class)
set.seed(1001)


bacc <- rep(0,length(threshold))
sens <- as.numeric(rep(0,length(threshold)))
spec <- as.numeric(rep(0,length(threshold)))

Attrition.train <- as.numeric(data.train$Attrition)
data.use1 <- data.use
data.use$BusinessTravel <- as.numeric(data.use$BusinessTravel)
data.use$Department <- as.numeric(data.use$Department)
data.use$EducationField <- as.numeric(data.use$EducationField)
data.use$EnvironmentSatisfaction <- as.numeric(data.use$EnvironmentSatisfaction)
data.use$Attrition <- as.numeric(data.use$Attrition)
data.use$Age <- as.numeric(data.use$Age)
data.use$DailyRate <- as.numeric(data.use$DailyRate)
data.use$DistanceFromHome <- as.numeric(data.use$DistanceFromHome)
data.use$MonthlyIncome <- as.numeric(data.use$MonthlyIncome)
data.use$NumCompaniesWorked <- as.numeric(data.use$NumCompaniesWorked )
data.use$TotalWorkingYears <- as.numeric(data.use$TotalWorkingYears)
data.use$TrainingTimesLastYear <- as.numeric(data.use$TrainingTimesLastYear)
data.use$YearsInCurrentRole <- as.numeric(data.use$YearsInCurrentRole)
data.use$YearsSinceLastPromotion <- as.numeric(data.use$YearsSinceLastPromotion)
data.use$YearsWithCurrManager <- as.numeric(data.use$YearsWithCurrManager)

set.seed(1001)
data.train <- data.use[train,]
data.test <- data.use[-train,]




train.x <- data.train[,-2]
test.x <- data.test[,-2]


for(i in 1:100){
  set.seed(1001)
  knn.pred <- knn(data.frame(train.x),data.frame(test.x),Attrition.train,k=i)
  sens[i] <- mean(knn.pred[data.test$Attrition==2]==2)
  spec[i] <- mean(knn.pred[data.test$Attrition==1]==1)
  bacc[i] <- .40*sens[i]+.60*spec[i]
}
plot(1:100,bacc,xlab="K value",ylab="Accuracy",type="l",main="K VS bacc")
which.max(bacc)

set.seed(1001)
knn.pred <- knn(data.frame(train.x),data.frame(test.x),Attrition.train,k=3)
cm <- table(data.test$Attrition,knn.pred)
spec <- cm[1,1]/(cm[1,1]+cm[1,2])
sens <- cm[2,2]/(cm[2,1]+cm[2,2])
accuracy <- (cm[1,1]+cm[2,2])/sum(cm)
bacc <-  .40*sens+.60*spec



###lda
data.use <- data.use1
set.seed(1001)
train <- sample(nrow(data.use),1100)
data.train <- data.use[train,]
data.test <- data.use[-train,]
sum(data.train$Attrition=='Yes')
sum(data.test$Attrition=='Yes')



library(MASS)

lda.fit <- lda(Attrition ~.,data=data.train)
lda.pred <- predict(lda.fit,data.test)

cm.lda<- table(data.test$Attrition,lda.pred$class)
accuracy <- (cm.lda[1,1]+cm.lda[2,2])/sum(cm.lda)
sens <- mean(lda.pred$class[data.test$Attrition=='Yes']=='Yes')
spec <- 1-mean(lda.pred$class[data.test$Attrition=='No']=='Yes')
bacc <-  .40*sens+.60*spec

###QDA

qda.fit <- qda(Attrition ~.,data=data.train)
qda.pred <- predict(qda.fit,data.test)

cm.qda <- table(data.test$Attrition,qda.pred$class)
accuracy <- (cm.qda[1,1]+cm.qda[2,2])/sum(cm.qda)
sens <- mean(qda.pred$class[data.test$Attrition=='Yes']=='Yes')
spec <- 1-mean(qda.pred$class[data.test$Attrition=="No"]=='Yes')
bacc <-  .40*sens+.60*spec

## bagging

library(randomForest)
set.seed(1001)


trees <- seq(5,200,by=10)
accuracy1 <- rep(0,length(trees))


for(i in 1:length(trees)){
  Bag.IBM=randomForest(data.train$Attrition ~ .,data=data.train,mtry=14,ntree=trees[i], importance=TRUE)
  yhat.bag=predict(Bag.IBM,newdata=data.test,type='class')
  cm.bag <- table(data.test$Attrition,yhat.bag)
  accuracy1[i] <- (cm.bag[1,1]+cm.bag[2,2])/sum(cm.bag)
}

trees[which.max(accuracy1)]

plot(trees,accuracy1,type='b',main='Bagging accuracy with different tree sizes')


set.seed(1001)
Bag.IBM=randomForest(data.train$Attrition~.,data=data.train,mtry=14,ntree=115, importance=TRUE)
yhat.bag=predict(Bag.IBM,newdata=data.test,type='class')
cm.bag <- table(data.test$Attrition,yhat.bag)
accuracy <- (cm.bag[1,1]+cm.bag[2,2])/sum(cm.bag)
spec <- cm.bag[1,1]/(cm.bag[1,1]+cm.bag[1,2])
sens <- cm.bag[2,2]/(cm.bag[2,1]+cm.bag[2,2])
bacc <-  .40*sens+.60*spec


## random forests

accuracy <- rep(0,8)
set.seed(1001)
for(i in 1:8){
  RF.ibm=randomForest(data.train$Attrition ~.,data=data.train,mtry=i,ntree=165, importance=TRUE)
  yhat.bag=predict(RF.ibm,newdata=data.test,type='class')
  cm.bag <- table(data.test$Attrition,yhat.bag)
  accuracy[i] <- (cm.bag[1,1]+cm.bag[2,2])/sum(cm.bag)
}

plot(1:8,accuracy,type='b',main='Accuracy for different ntree')
which.max(accuracy)

## for 3 tree depth select no of trees
trees <- seq(5,200,by=10)
accuracy <- rep(0,length(trees))

set.seed(1001)
for(i in 1:length(trees)){
  RF.ibm=randomForest(data.train$Attrition~.,data=data.train,mtry=3,ntree=trees[i], importance=TRUE)
  yhat.bag=predict(RF.ibm,newdata=data.test,type='class')
  cm.bag <- table(data.test$Attrition,yhat.bag)
  accuracy[i] <- (cm.bag[1,1]+cm.bag[2,2])/sum(cm.bag)
}

plot(trees,accuracy,type='b',main='accuracy with different tree sizes for tree depth = 3')
trees[which.max(accuracy)]

set.seed(1001)
RF.ibm=randomForest(data.train$Attrition ~.,data=data.train,mtry=3,ntree=115, importance=TRUE)
yhat.bag=predict(RF.ibm,newdata=data.test,type='class')
cm.bag <- table(data.test$Attrition,yhat.bag)
accuracy <- (cm.bag[1,1]+cm.bag[2,2])/sum(cm.bag)
spec <- cm.bag[1,1]/(cm.bag[1,1]+cm.bag[1,2])
sens <- cm.bag[2,2]/(cm.bag[2,1]+cm.bag[2,2])
bacc <-  .40*sens+.60*spec

##svm with linear kernel

library(e1071)

set.seed(1001)
tune.out <- tune(svm,Attrition~.,data=data.train,kernel='linear',ranges=list(cost=c(.01,.1,.5,.10,.50,1,5,10)))
summary(tune.out)
bestmod1 <- tune.out$best.model
summary(bestmod1)

set.seed(1081)
svmfit1 <- svm(Attrition~.,data=data.train,kernel="linear",cost=0.1,scale=FALSE)
svm.pred1 <- predict(svmfit1,data.test,type="class")
cm.svm <- table(data.test$Attrition,svm.pred1)
accuracy <- (cm.svm[1,1]+cm.svm[2,2])/(sum(cm.svm))
spec <- cm.svm[1,1]/(cm.svm[1,1]+cm.svm[1,2])
sens <- cm.svm[2,2]/(cm.svm[2,1]+cm.svm[2,2])
bacc <-  .40*sens+.60*spec

##svm with radial kernel
set.seed(1091)
svm.tun.rad <- tune(svm,Attrition~.,data=data.train,kernel='radial',ranges=list(cost=1,gamma=c(0.5,1,2,3)))
bestmod.rad <- svm.tun.rad$best.model
summary(bestmod.rad)

svm.pred.radb <- predict(svm.tun.rad$best.model,data.test,type='class')
cm.svm <- table(data.test$Attrition,svm.pred.radb)
accuracy <- (cm.svm[1,1]+cm.svm[2,2])/(sum(cm.svm))
spec <- cm.svm[1,1]/(cm.svm[1,1]+cm.svm[1,2])
sens <- cm.svm[2,2]/(cm.svm[2,1]+cm.svm[2,2])
bacc <-  .40*sens+.60*spec

##svm with polynomial kernel

set.seed(1001)
svm.tun.poly <- tune(svm, Attrition~.,data=data.train,kernel='polynomial',ranges=list(cost=c(0.1,1,10,100),degree=c(2,3,4,5)))
summary(svm.tun.poly)
bestmod.tun.poly <- svm.tun.poly$best.model

svm.pred.polyb <- predict(svm.tun.poly$best.model,data.test,type='class')
cm.svm <- table(data.test$Attrition,svm.pred.polyb)
accuracy <- (cm.svm[1,1]+cm.svm[2,2])/(sum(cm.svm))
spec <- cm.svm[1,1]/(cm.svm[1,1]+cm.svm[1,2])
sens <- cm.svm[2,2]/(cm.svm[2,1]+cm.svm[2,2])
bacc <-  .40*sens+.60*spec

# boosting
library(gbm)
set.seed(1001)
boost.attrition <- gbm(Attrition~.,data = data.train,distribution = 'multinomial',n.trees=5000,interaction.depth = 6,shrinkage = .005,cv.folds = 10)
pred.boost <- predict(boost.attrition,data.test,n.trees=5000)
pred1.boost <- apply(pred.boost, 1, which.max)
pred1.boost[pred1.boost==1] <- 0
pred1.boost[pred1.boost==2] <- 1
cm.boost <- table(data.test$Attrition,pred1.boost)
accuracy <- (cm.boost[1,1]+cm.boost[2,2])/(sum(cm.boost))
spec <- cm.boost[1,1]/(cm.boost[1,1]+cm.boost[1,2])
sens <- cm.boost[2,2]/(cm.boost[2,1]+cm.boost[2,2])
bacc <-  .40*sens+.60*spec

detach(data.use)
## boosting with entire data
attach(mydata)
set.seed(1001)
subset.data <- cbind(subset.data[,2],subset.data[,-2])
colnames(subset.data)[1] <- 'Attrition'

train <- sample(nrow(mydata),1100)
data.train <- subset.data[train,]
data.test <- subset.data[-train,]
sum(data.train$Attrition=='Yes')
sum(data.test$Attrition=='Yes')

boost.attrition <- gbm(Attrition~.,data = data.train,distribution = 'multinomial',n.trees=5000,interaction.depth = 6,shrinkage = .005,cv.folds = 10)
pred.boost <- predict(boost.attrition,data.test,n.trees=5000)
pred1.boost <- apply(pred.boost, 1, which.max)
pred1.boost[pred1.boost==1] <- 0
pred1.boost[pred1.boost==2] <- 1
cm.boost <- table(data.test$Attrition,pred1.boost)
accuracy <- (cm.boost[1,1]+cm.boost[2,2])/(sum(cm.boost))
spec <- cm.boost[1,1]/(cm.boost[1,1]+cm.boost[1,2])
sens <- cm.boost[2,2]/(cm.boost[2,1]+cm.boost[2,2])
bacc <-  .40*sens+.60*spec


detach(mydata)
### xtreme gradient boosting

library(xgboost)
library(data.table)
library(Matrix)
train= data.table(data.train) #convert train set to data table format
test= data.table(data.test) #convert test set to data table format
sparse.matrix.train= sparse.model.matrix(Attrition ~.-1, data = train) #converts train set factors to columns
sparse.matrix.test=  sparse.model.matrix(Attrition ~.-1, data = test) #converts test set factors to columns

label <- rep(0,1100)
label[data.train$Attrition=='Yes'] <- 1
label <- as.numeric(label)


set.seed(1001)
xgb.IBM <- xgboost(data = sparse.matrix.train,
                   label= label,
                   eta= .3,
                   max_depth=8,
                   subsample= 1,
                   colsample_bytree=.5,
                   seed=1001,
                   objective= 'binary:logistic',
                   eval_metric = 'logloss',
                   nrounds = 5000,
                   nthread=3,
                   scale_pos_weight=1,
                   min_child_weight=.5)

pred.xgb <- predict(xgb.IBM,sparse.matrix.test)
prediction <- as.numeric(pred.xgb > 0.5)
pred.xgb[prediction == 1] <- 'Yes'
pred.xgb[prediction == 0] <- 'No'







cm.bag <- table(data.test$Attrition,pred.xgb)
accuracy <- (cm.bag[1,1]+cm.bag[2,2])/sum(cm.bag)
spec <- cm.bag[1,1]/(cm.bag[1,1]+cm.bag[1,2])
sens <- cm.bag[2,2]/(cm.bag[2,1]+cm.bag[2,2])
bacc <-  .40*sens+.60*spec
  
## smote
library(ROSE)
data.trainxgb <- ROSE(Attrition ~ ., data = data.train, seed = 1012)$data
data.test$Attrition <- as.numeric(data.test$Attrition)


label <- rep(0,1100)
label[data.trainxgb$Attrition== 'Yes'] <- 1
label <- as.numeric(label)


library(xgboost)
library(data.table)
library(Matrix)
train= data.table(data.trainxgb) #convert train set to data table format
test= data.table(data.test) #convert test set to data table format
sparse.matrix.train= sparse.model.matrix(Attrition ~.-1, data = train) #converts train set factors to columns
sparse.matrix.test=  sparse.model.matrix(Attrition ~.-1, data = test) #converts test set factors to columns


set.seed(999)
xgb.IBM <- xgboost(data = sparse.matrix.train,
                   label= label,
                   eta= .2,
                   max_depth=7,
                   subsample= .5,
                   colsample_bytree=.5,
                   seed=999,
                   objective= 'binary:logistic',
                   eval_metric = 'logloss',
                   nrounds = 10000,
                   nthread=3)

pred.xgb <- predict(xgb.IBM,sparse.matrix.test)
prediction <- as.numeric(pred.xgb > 0.5)
pred.xgb[prediction == 1] <- "Yes"
pred.xgb[prediction == 0] <- "No"


cm.bag <- table(data.test$Attrition,pred.xgb)
accuracy <- (cm.bag[1,1]+cm.bag[2,2])/sum(cm.bag)
spec <- cm.bag[1,1]/(cm.bag[1,1]+cm.bag[1,2])
sens <- cm.bag[2,2]/(cm.bag[2,1]+cm.bag[2,2])
bacc <-  .40*sens+.60*spec

### neural nets

data.train$Attrition <- as.numeric(data.train$Attrition)-1
m <- model.matrix(~ Attrition+Age+BusinessTravel+DailyRate+Department+DistanceFromHome+EducationField+EnvironmentSatisfaction+MonthlyIncome+NumCompaniesWorked+TotalWorkingYears+TrainingTimesLastYear+YearsInCurrentRole+YearsSinceLastPromotion+YearsWithCurrManager,data=data.train)
colnames(m)[7] <- 'DepartmentResearch_Development'
colnames(m)[10] <- 'EducationFieldLife_Sciences'
colnames(m)[14] <- 'EducationFieldTechnical_Degree'

data.test$Attrition <- as.numeric(data.test$Attrition)-1
m.test <- m <- model.matrix(~ Attrition+Age+BusinessTravel+DailyRate+Department+DistanceFromHome+EducationField+EnvironmentSatisfaction+MonthlyIncome+NumCompaniesWorked+TotalWorkingYears+TrainingTimesLastYear+YearsInCurrentRole+YearsSinceLastPromotion+YearsWithCurrManager,data=data.test)
colnames(m)[7] <- 'DepartmentResearch_Development'
colnames(m)[10] <- 'EducationFieldLife_Sciences'
colnames(m)[14] <- 'EducationFieldTechnical_Degree'


library(neuralnet)

#scaled.data <- as.data.frame(scale(,center = mins, scale = maxs - mins))


mod.net <- neuralnet(Attrition ~Age+BusinessTravelTravel_Frequently+BusinessTravelTravel_Rarely+DailyRate+DepartmentResearch_Development+DepartmentSales+DistanceFromHome+EducationFieldLife_Sciences+EducationFieldMarketing+EducationFieldMedical+EducationFieldOther+EducationFieldTechnical_Degree +EnvironmentSatisfaction2+EnvironmentSatisfaction3+EnvironmentSatisfaction4+MonthlyIncome+NumCompaniesWorked+TotalWorkingYears+TrainingTimesLastYear+YearsInCurrentRole+YearsSinceLastPromotion+YearsWithCurrManager, data=m[,-1],hidden = 4,lifesign = 'minimal',linear.output = F,threshold = .1)
pred.nn <- compute(mod.net,m.test[,-c(1,2)])



### c 50
library(C50)


c50.mod <- C5.0(Attrition ~.,data=data.train,rules=T,trials=100)
pred.c50 <- predict(c50.mod,newdata = data.test,type='class')

cm.bag <- table(data.test$Attrition,pred.c50)
accuracy <- (cm.bag[1,1]+cm.bag[2,2])/sum(cm.bag)
spec <- cm.bag[1,1]/(cm.bag[1,1]+cm.bag[1,2])
sens <- cm.bag[2,2]/(cm.bag[2,1]+cm.bag[2,2])
bacc <-  .40*sens+.60*spec



