#==============================================================================
###          KNN MIXT SENSE SMOTE SENSE VARS IRRELLEVANTS                  ####
#==============================================================================

# seleccionem les variables que surten a les association rules
vars <- c("ID","Age","IsActiveMember","CreditScore","NumOfProducts","MaritalStatus",
          "Gender","SavingsAccountFlag","TransactionFrequency","LoanStatus",
          "Balance","NetPromoterScore","ComplaintsCount","AvgTransactionAmount")

vars_ex <- c("Age","IsActiveMember","CreditScore","NumOfProducts","MaritalStatus",
             "Gender","SavingsAccountFlag","TransactionFrequency","LoanStatus",
             "Balance","NetPromoterScore","ComplaintsCount","AvgTransactionAmount","Exited")


load("data_NA_imputed_AREG_test.RData")
load("dataAREG_outliers.RData")

f1 <- function(data, lev = NULL, model = NULL) {
  f1_val <- MLmetrics::F1_Score(y_pred = data$pred, y_true = data$obs, positive = "Yes")
  c(F1 = f1_val)
}


### Preparem dades del test ==================================================

data_test <- data_imputed_AREG_test[-6] #eliminem surname
data_test <- data_test[,vars]

clases_test <- sapply(data_test, class)
varCat_test <- names(clases_test)[which(clases_test %in% c("character", "factor"))]
varNum_test <- names(clases_test)[which(clases_test %in% c("numeric", "integer"))]

datanum_test <- data_test[,varNum_test]
dataCat_test<- data_test[,varCat_test]

data_dummies_test <- dummy_cols(dataCat_test, remove_first_dummy = TRUE, 
                                remove_selected_columns = TRUE)

data_mixta_test <- cbind(datanum_test, data_dummies_test)

### Preparem dades train =======================================================
data <- dataAREG
data <- dataAREG[,vars_ex]

clases <- sapply(data, class)
varNum <- names(clases)[which(clases %in% c("numeric", "integer"))]
varCat <- names(clases)[which(clases %in% c("character", "factor"))]

datanum <- data[,varNum]
datacat <- data[,varCat]


dataCat<- data[varCat][-8] # fora Exited, no volem que faci dummies
data_dummies <- dummy_cols(dataCat, remove_first_dummy = TRUE, 
                           remove_selected_columns = TRUE)

Exited <- data$Exited
data_mixta <- cbind(datanum, data_dummies, Exited)

### separem train i test ======================================================

set.seed(123)

ind_col <- c(23) # columna on està exited
default_idx <- createDataPartition(data_mixta$Exited, p = 0.8, list = FALSE)
X_trainC <- data_mixta[default_idx, ]
X_testC <- data_mixta[-default_idx, ]
y_testC <- X_testC[, ind_col]
X_testC <- X_testC[, -ind_col]

X_trainC$Exited <- factor(X_trainC$Exited, levels = c(0,1), labels = c("No", "Yes"))
y_testC  <- factor(y_testC,  levels = c(0,1), labels = c("No", "Yes"))

#preparem les dades
preproc <- preProcess(X_trainC, method = c("center", "scale"))
X_trainC <- predict(preproc, X_trainC)
X_testC <- predict(preproc, X_testC)


ctrl <- trainControl(method="repeatedcv",
                     number= 5,
                     repeats = 3,
                     classProbs=TRUE,
                     summaryFunction = f1)

grid <- expand.grid(k = seq(1, 11, by = 2))

knn_mixto <- train(
  Exited ~ ., 
  data = X_trainC, 
  method = "knn", 
  trControl = ctrl, 
  tuneGrid = grid,
  metric = "F1")
knn_mixto

plot(knn_mixto)

get_best_result = function(caret_fit) {
  best = which(rownames(caret_fit$results) == rownames(caret_fit$bestTune))
  best_result = caret_fit$results[best, ]
  rownames(best_result) = NULL
  best_result
}

best_model_mixto <- get_best_result(knn_mixto) # una altra vegada k=1

predictions <-predict(knn_mixto, newdata = X_testC, type = "prob")

cm_1 <- confusionMatrix(predict(knn_mixto, newdata = X_testC), y_testC,positive="Yes")
cm_1

roc(y_testC, as.data.frame(predictions)[,"Yes"])

roc_obj <- roc(y_testC, 
               predictions[,"Yes"], 
               plot=TRUE, 
               print.auc=TRUE, 
               legacy.axes=FALSE)

# extraiem totes les mètriques per després fer una taula comparativa:
tp <- cm_1$table[2,2]
fp <- cm_1$table[2,1]
fn <- cm_1$table[1,2]

precision <- tp/(tp+fp)
recall_1 <- tp/(tp+fn)
f1_1 <- 2*precision*recall_1/(precision+recall_1)
Accuracy_1 = cm_1$overall["Accuracy"]
Sensitivity_1 = cm_1$byClass["Sensitivity"]
Specificity_1 = cm_1$byClass["Specificity"]

auc_value_1 <- auc(roc_obj)


# prediccions per test (submission)
pred_test <- predict(knn_mixto, newdata = data_mixta_test, type = "raw")

submission_knn1 <- data.frame(
  ID = data_test$ID,
  Exited = pred_test
)

write.csv(submission_knn1, "submission_knn1_seleccion.csv", row.names = FALSE)


#==============================================================================
###                             KNN MIXT K=3                               ####
#==============================================================================

grid <- expand.grid(k = 3)

knn_mixto_3 <- train(
  Exited ~ ., 
  data = X_trainC, 
  method = "knn", 
  trControl = ctrl, 
  tuneGrid = grid,
  metric = "F1")


best_model_mixto_3 <- get_best_result(knn_mixto_3) # una altra vegada k=1

predictions <-predict(knn_mixto_3, newdata = X_testC, type = "prob")

cm_3 <- confusionMatrix(predict(knn_mixto_3, newdata = X_testC), y_testC,positive="Yes")
cm_3

roc(y_testC, as.data.frame(predictions)[,"Yes"])

roc_obj <- roc(y_testC, 
               predictions[,"Yes"], 
               plot=TRUE, 
               print.auc=TRUE, 
               legacy.axes=FALSE)

# extraiem totes les mètriques per després fer una taula comparativa:
tp <- cm_3$table[2,2]
fp <- cm_3$table[2,1]
fn <- cm_3$table[1,2]

precision <- tp/(tp+fp)
recall_3 <- tp/(tp+fn)
f1_3 <- 2*precision*recall_3/(precision+recall_3)
Accuracy_3 = cm_3$overall["Accuracy"]
Sensitivity_3 = cm_3$byClass["Sensitivity"]
Specificity_3 = cm_3$byClass["Specificity"]

auc_value_3 <- auc(roc_obj)


# prediccions per test (submission)
pred_test <- predict(knn_mixto_3, newdata = data_mixta_test, type = "raw")

submission_knn3 <- data.frame(
  ID = data_test$ID,
  Exited = pred_test
)

write.csv(submission_knn3, "submission_knn3_seleccion.csv", row.names = FALSE)

resultats <- data.frame(
  Model = c("k=1", "k=3"),
  F1 = c(f1_1, f1_3),
  Accuracy = c(Accuracy_1, Accuracy_3),
  Sensitivity = c(Sensitivity_1, Sensitivity_3),
  Specificity = c(Specificity_1, Specificity_3),
  AUC = c(auc_value_1, auc_value_3),
  Recall = c(recall_1, recall_3)
)

print(resultats)

save(best_model_mixto, best_model_mixto_3, knn_mixto, knn_mixto_3, resultats,
     file = "models_knn_selec.RData")


