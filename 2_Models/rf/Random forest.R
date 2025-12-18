###MAIN PACKAGES
library(DAAG)
library(mlbench)
library(caret)
library(pROC)
library(printr)
library(randomForest)
library(ranger)


load("dataAREG_outliers.RData")

mydata <- dataAREG
set.seed(123)
ind <- sample(1:nrow(mydata), 0.8*nrow(mydata))
train <- mydata[ind,]
test <- mydata[-ind,]

train$Exited <- factor(train$Exited, levels = c(0,1), labels = c("No", "Yes"))
test$Exited  <- factor(test$Exited,  levels = c(0,1), labels = c("No", "Yes"))

####Modelling with RandomForest

###Bagging trees, mtry=p, muy simple

bagtrees <- randomForest(Exited ~ ., data = train, mtry = ncol(train) - 1)
bagtrees # matriz confusion en el train
plot(bagtrees, main = "")
legend("right", colnames(bagtrees$err.rate), lty = 1:5, col = 1:6)
pred2 <- predict(bagtrees, newdata = test)
caret::confusionMatrix(pred2, test$Exited, positive="Yes")

# Accuracy del 0.8006 y Sensitivity : 0.28136 
# (poner threshold, algo tiene q mejorar)

###Random Forest

rf <- randomForest(Exited ~ ., data = train)
rf
plot(rf,main="")
legend("right", colnames(rf$err.rate), lty = 1:5, col = 1:6)
importance(rf) # no funciona
varImpPlot(rf)
pred3 <- predict(rf, newdata = test)
caret::confusionMatrix(pred3, test$Exited,positive="Yes")

# Accuracy 0.807, Sensitivity : 0.22034

####Random Forest with caret 
rf.caret <- train(Exited ~ ., data = train, method = "rf")
plot(rf.caret)
pred3a <- predict(rf.caret, newdata = test)
caret::confusionMatrix(pred3a, test$Exited,positive="Yes") ###Best performance so far

# mirar si podem fer q escolleixi millor model per sensitivity, no accuracy

##mtry could be fitted by using mtry by default, mrty/2 and 2*mtry. Remember mrty by default is sqrt(p)-->classification and p/3 for regression problems.
### Also ntree could be fitted by "playing" with different values for it.

mtry.class <- sqrt(ncol(train) - 1)
tuneGrid <- data.frame(mtry = floor(c(mtry.class/2, mtry.class, 2*mtry.class)))
tuneGrid
set.seed(1234)
rf.caret <- train(Exited ~ ., data = train,method = "rf",tuneGrid = tuneGrid)
plot(rf.caret)
rf.caret
pred4 <- predict(rf.caret, newdata = test)
caret::confusionMatrix(pred4, test$Exited,positive="Yes") 

# saber si rf tiene overfitting -> calculamos con train y comparamos con test

ptrain <- predict(rf.caret, train, type = 'raw')
confusionMatrix(ptrain, train$Exited, positive="Yes")
confusionMatrix(pred4, test$Exited, positive="Yes")


                    

# threshold tiene q ser 0.2
# mirar codigo trees que está como hacerlo y dejarlo clasificado

obs <- test$Exited
caret::postResample(pred4, obs)