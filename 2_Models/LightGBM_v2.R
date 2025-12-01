#library(pacman)
#pacman::p_load(pscl, ggplot2, ROCR, lightgbm, methods, Matrix, caret, dplyr)

set.seed(123)

load("data_Final_AREG_train.RData")
data <- dataAREG_final

data$ID <- NULL
data$Surname <- NULL
data$Exited <- as.numeric(as.character(data$Exited))

Index <- createDataPartition(data$Exited, p = 0.7, list = FALSE)
dataTrain <- data[Index, ]
dataTest  <- data[-Index, ]


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
#plot(roc,main="ROC curve")
#abline(a=0,b=1)
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



################################################################################
#####                            THRESHOLD                                ####
################################################################################

thresholds <- seq(0.01, 0.8, by = 0.01)

# Inicializar un vector para almacenar los resultados del F1
f1_scores <- numeric(length(thresholds))

# Loop para probar cada threshold y calcular el F1
for (i in seq_along(thresholds)) {
  threshold <- thresholds[i]
  preds <- ifelse(p > threshold, 1, 0)
  f1_scores[i] <- MLmetrics::F1_Score(y_pred = preds, y_true = dataTest$Exited,
                                      positive = 1)
}

threshold_f1_df <- data.frame(Threshold = thresholds, F1_Score = f1_scores)

print(threshold_f1_df)

best_threshold <- threshold_f1_df$Threshold[which.max(threshold_f1_df$F1_Score)]
best_threshold

preds_testtrain <- ifelse(p > best_threshold, 1, 0)

################################################################################
#####                            Para Kaggle                                ####
################################################################################

load("data_Final_AREG_test.RData")

data2 <- dataAREG_test_final

data2ID <- data2$ID
data2$ID <- NULL
data2$Surname <- NULL

test_m2 <- sparse.model.matrix( ~., data = data2)

# 1. Obtener probabilidades de test
probs_test <- predict(bst, newdata = test_m2)

# 2. Aplicar el mejor threshold encontrado
best_threshold <- threshold_f1_df$Threshold[which.max(threshold_f1_df$F1_Score)]
pred_test <- ifelse(probs_test > best_threshold, "Yes", "No")


# 3. Crear el dataframe de submission
submission <- data.frame(
  ID = data2ID,
  Exited = pred_test
)

# 4. Guardar el CSV
write.csv(submission, "submission_LightGBM_v2_01.csv", row.names = FALSE)
