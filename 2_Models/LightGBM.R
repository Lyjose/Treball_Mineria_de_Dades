# library(pacman)
# pacman::p_load(pscl, ggplot2, ROCR, lightgbm, methods, Matrix, caret, dplyr)

set.seed(123)

load("dataAREG_outliers.RData")
data <- dataAREG

data$DigitalEngagementScore <- factor(data$DigitalEngagementScore)
data$CustomerSegment <- factor(data$CustomerSegment)
data$NetPromoterScore <- factor(data$NetPromoterScore)
data$Exited <- as.numeric(as.character(data$Exited))

Index <- createDataPartition(data$Exited, p = 0.8, list = FALSE)
dataTrain <- data[Index, ]
dataTest  <- data[-Index, ]

seleccio = c("Age","IsActiveMember","MaritalStatus","EstimatedSalary", 
             "SavingsAccountFlag","NumOfProducts","Gender","AvgTransactionAmount",
             "Geography","EducationLevel","HasCrCard", "Exited")
dataTrain <- dataTrain %>% 
  select(seleccio)
dataTest <- dataTest %>% 
  select(seleccio)

train_m <- sparse.model.matrix(Exited ~., data = dataTrain)
train_label <- dataTrain$Exited

test_m <- sparse.model.matrix(Exited ~., data = dataTest)
test_label <- dataTest$Exited

train_matrix <- lgb.Dataset(data = as.matrix(train_m), label = train_label)
test_matrix <- lgb.Dataset(data = as.matrix(test_m), label = test_label)



valid <- list(test = test_matrix)

# model parameters
params <- list(max_bin = 5,
              learning_rate = 0.001,
              objective = "binary",
              metric = 'binary_logloss')


bst <- lightgbm(params = params, data = train_matrix, valids = list(test = test_matrix), nrounds = 1000)

#prediction & confusion matrix
p <- predict(bst, test_m)
dataTest$Predicted <- ifelse(p > 0.2, 1, 0)
confusionMatrix(factor(dataTest$Predicted), factor(dataTest$Exited))


pred <- prediction(p,dataTest$Exited)
roc <- performance(pred,"tpr","fpr")
auc <- performance(pred, measure = "auc")
plot(roc,main="ROC curve")
abline(a=0,b=1)
auc@y.values


f1 <- function(data, lev = NULL, model = NULL) {
  f1_val <- MLmetrics::F1_Score(y_pred = data$pred, y_true = data$obs, positive = 1)
  c(F1 = f1_val)
}

results <- data.frame(
  pred = dataTest$Predicted,
  obs  = dataTest$Exited
)

(f1_val <- f1(data = results))








