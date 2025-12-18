# ==============================================================================
# KNN
# ==============================================================================


packages <- c("ISLR", "caret", "vcd", "pROC", "VIM","fastDummies","themis","MLmetrics")

install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

lapply(packages, install_if_missing)


### Preparació de les dades ####

load("data_NA_imputed_AREG_test.RData")
load("dataAREG_outliers.RData")

data_test <- data_imputed_AREG_test[-6]
data <- dataAREG

clases <- sapply(data, class)


varNum <- names(clases)[which(clases %in% c("numeric", "integer"))]
varCat <- names(clases)[which(clases %in% c("character", "factor"))]

datanum <- data[,varNum]
datacat <- data[,varCat]



# Divisió entre train i test -> 80% train 20% test

set.seed(123)
muestra <- sample(1:nrow(datanum), size = nrow(datanum)/5)
cl<-data$Exited[-muestra]
data_KNN <- data.frame(datanum, Exited = data$Exited)

# Posem funció f1


f1 <- function(data, lev = NULL, model = NULL) {
  f1_val <- MLmetrics::F1_Score(y_pred = data$pred, y_true = data$obs, positive = "Yes")
  c(F1 = f1_val)
}



train <- data_KNN[-muestra, ]
test <- data_KNN[muestra, ]

train$Exited <- factor(train$Exited, levels = c(0,1), labels = c("No", "Yes"))
test$Exited  <- factor(test$Exited,  levels = c(0,1), labels = c("No", "Yes"))


preProcValues <- preProcess(train, method = c("center", "scale")) # estandardització de les dades
trainTransformed <- predict(preProcValues, train)
testTransformed <- predict(preProcValues, test)


#=============================================================================#
###                                KNN sin SMOTE (k=3                      ####
#=============================================================================#

ctrl <- trainControl(method="repeatedcv",
                     number= 5,
                     repeats = 3,
                     classProbs=TRUE,
                     summaryFunction = f1)

grid <- data.frame(k = c(3,5,7,9,11,13))

knnModel <- train(
  Exited ~ ., 
  data = trainTransformed, 
  method = "knn", 
  trControl = ctrl, 
  tuneGrid = grid,
  metric = "F1")

knnModel
plot(knnModel)

best_model <- knn3(
  Exited ~ .,
  data = trainTransformed,
  k = knnModel$bestTune$k
)
best_model

predictions <- predict(best_model, testTransformed,type = "class")

# Fem matriu de confusió
cm <- confusionMatrix(predictions, testTransformed$Exited, positive = "Yes")
cm

pred_probs <- predict(best_model, testTransformed, type = "prob")
roc(testTransformed$Exited, as.data.frame(pred_probs)[,"Yes"])

roc_obj <- roc(testTransformed$Exited, 
               pred_probs[,"Yes"], 
               plot=TRUE, 
               print.auc=TRUE, 
               legacy.axes=FALSE)

# extraiem totes les mètriques per després fer una taula comparativa:
tp <- cm$table[2,2]
fp <- cm$table[2,1]
fn <- cm$table[1,2]

precision <- tp/(tp+fp)
recall_1 <- tp/(tp+fn)
f1_1 <- 2*precision*recall_1/(precision+recall_1)
Accuracy_1 = cm$overall["Accuracy"]
Sensitivity_1 = cm$byClass["Sensitivity"]
Specificity_1 = cm$byClass["Specificity"]

auc_value_1 <- auc(roc_obj)

# k=3
# és possible que k=1 sigui millor que k=3

#=============================================================================#
###                                KNN sin SMOTE (k=1)                     ####
#=============================================================================#

grid <- data.frame(k = 1)

knnModel_1 <- train(
  Exited ~ ., 
  data = trainTransformed, 
  method = "knn", 
  trControl = ctrl, 
  tuneGrid = grid,
  metric = "F1")

knnModel_1

best_model_1 <- knn3(
  Exited ~ .,
  data = trainTransformed,
  k = knnModel_1$bestTune$k
)
best_model_1 # el millor model és amb k=1, (k=11 quan feiem amb accuracy)

predictions <- predict(best_model_1, testTransformed,type = "class")

# Fem matriu de confusió
cm_2 <- confusionMatrix(predictions, testTransformed$Exited, positive = "Yes")
cm_2

# volem més Sensitivity perquè volem que controli bé els positius. No volem falsos negatius
# és a dir, que predigui que no marxaran però marxen.

tp <- cm_2$table[2,2]
fp <- cm_2$table[2,1]
fn <- cm_2$table[1,2]

precision <- tp/(tp+fp)
recall_2 <- tp/(tp+fn)
f1_2 <- 2*precision*recall_2/(precision+recall_2)
Accuracy_2 = cm_2$overall["Accuracy"]
Sensitivity_2 = cm_2$byClass["Sensitivity"]
Specificity_2 = cm_2$byClass["Specificity"]

auc_value_2 <- auc(roc_obj)




#=============================================================================#
###                               KNN MIXTO (k=1)                          ####
#=============================================================================#

# fem dummies
dataCat<- data[varCat][-12]
data_dummies <- dummy_cols(dataCat, remove_first_dummy = TRUE, 
                           remove_selected_columns = TRUE)

Exited <- data$Exited
data_mixta <- cbind(data[, varNum], data_dummies, Exited)

# preparamos para aplicar el test

clases_test <- sapply(data_test, class)
varCat_test <- names(clases_test)[which(clases_test %in% c("character", "factor"))]
varNum_test <- names(clases_test)[which(clases_test %in% c("numeric", "integer"))]

datanum_test <- data_test[,varNum_test]
dataCat_test<- data_test[,varCat_test]

data_dummies_test <- dummy_cols(dataCat_test, remove_first_dummy = TRUE, 
                           remove_selected_columns = TRUE)

data_mixta_test <- cbind(datanum_test, data_dummies_test)


set.seed(123)

# separem les dades



ind_col <- c(34) # columna on està exited
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

cm_3 <- confusionMatrix(predict(knn_mixto, newdata = X_testC), y_testC,positive="Yes")
cm_3

roc(y_testC, as.data.frame(pred_probs)[,"Yes"])

roc_obj <- roc(y_testC, 
               pred_probs[,"Yes"], 
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
pred_test <- predict(knn_mixto, newdata = data_mixta_test, type = "raw")

submission_knn1 <- data.frame(
  ID = data_test$ID,
  Exited = pred_test
)

write.csv(submission_knn1, "submission_knn1_nosmote.csv", row.names = FALSE)


#=============================================================================#
###                               KNN MIXTO (k=3)                          ####
#=============================================================================#

grid <- expand.grid(k = 3)

knn_mixto_k3 <- train(
  Exited ~ ., 
  data = X_trainC, 
  method = "knn", 
  trControl = ctrl, 
  tuneGrid = grid,
  metric = "F1")
knn_mixto


get_best_result = function(caret_fit) {
  best = which(rownames(caret_fit$results) == rownames(caret_fit$bestTune))
  best_result = caret_fit$results[best, ]
  rownames(best_result) = NULL
  best_result
}

best_model_mixto <- get_best_result(knn_mixto_k3) # una altra vegada k=1

predictions <-predict(knn_mixto_k3, newdata = X_testC, type = "prob")

cm_4 <- confusionMatrix(predict(knn_mixto_k3, newdata = X_testC), y_testC,positive="Yes")
cm_4

roc(y_testC, as.data.frame(pred_probs)[,"Yes"])

roc_obj <- roc(y_testC, 
               pred_probs[,"Yes"], 
               plot=TRUE, 
               print.auc=TRUE, 
               legacy.axes=FALSE)

# extraiem totes les mètriques per després fer una taula comparativa:
tp <- cm_4$table[2,2]
fp <- cm_4$table[2,1]
fn <- cm_4$table[1,2]

precision <- tp/(tp+fp)
recall_4 <- tp/(tp+fn)
f1_4 <- 2*precision*recall_4/(precision+recall_4)
Accuracy_4 = cm_4$overall["Accuracy"]
Sensitivity_4 = cm_4$byClass["Sensitivity"]
Specificity_4 = cm_4$byClass["Specificity"]

auc_value_4 <- auc(roc_obj)

pred_test <- predict(knn_mixto_k3, newdata = data_mixta_test, type = "raw")

submission_knn3 <- data.frame(
  ID = data_test$ID,
  Exited = pred_test
)

write.csv(submission_knn3, "submission_knn3_nosmote.csv", row.names = FALSE)


#==============================================================================
###                          KNN MIXT AMB SMOTE                            ####
#==============================================================================

#cal utilitzar data_mixta

set.seed(123)
ind_col <- c(34)

# separem les dades en train i test

index <- createDataPartition(data_mixta$Exited, p = 0.8, list = FALSE)
train_data <- data_KNN[index, ]
test_data  <- data_KNN[-index, ]

train_data$Exited <- factor(train_data$Exited, levels = c(0,1), labels = c("No", "Yes"))
test_data$Exited  <- factor(test_data$Exited,  levels = c(0,1), labels = c("No", "Yes"))

# estandardització de les dades
preproc <- preProcess(train_data, method = c("center", "scale")) 
trainTransformed <- predict(preproc, train_data)
testTransformed <- predict(preproc, test_data)


ctrl <- trainControl(method="repeatedcv",
                     number= 5,
                     repeats = 3,
                     classProbs=TRUE,
                     summaryFunction = f1,
                     sampling = "smote")

grid <- expand.grid(k = seq(3, 39, by = 2))


set.seed(123)
model_smote <- caret::train(Exited ~ .,
                               data = trainTransformed,
                               method = "knn",
                               trControl = ctrl,
                               tuneGrid = grid,
                               metric = "F1"
                               )

plot(model_smote) 

best_model_smote <- knn3(
  Exited ~ .,
  data = trainTransformed,
  k = model_smote$bestTune$k
)
best_model_smote # k=29


predictions <- predict(best_model_smote, testTransformed,type = "class")

# Fem matriu de confusió
cm_5 <- confusionMatrix(predictions, testTransformed$Exited, positive = "Yes")
cm_5

# FATAL. Accuracy bé però Sensitivity : 0.02083
# Prediu molt que no marxaran i té molts falsos negatius. No és el que volem.

pred_probs <- predict(best_model_smote, testTransformed, type = "prob")
roc(testTransformed$Exited, as.data.frame(pred_probs)[,"Yes"])

roc_obj <- roc(testTransformed$Exited, 
               pred_probs[,"Yes"], 
               plot=TRUE, 
               print.auc=TRUE, 
               legacy.axes=FALSE)

# extraiem totes les mètriques per després fer una taula comparativa:
tp <- cm_5$table[2,2]
fp <- cm_5$table[2,1]
fn <- cm_5$table[1,2]

precision <- tp/(tp+fp)
recall_5 <- tp/(tp+fn)
f1_5 <- 2*precision*recall_5/(precision+recall_5)
Accuracy_5 = cm_5$overall["Accuracy"]
Sensitivity_5 = cm_5$byClass["Sensitivity"]
Specificity_5 = cm_5$byClass["Specificity"]

auc_value_5 <- auc(roc_obj)

# en el plot diu que el F1 ha de ser de 0.38, però en canvi al passar el test
# surt de 0.03, això és perquè no acerta els positius.

# Això indica un fort overfitting del train.



#==============================================================================
###                                RESULTATS                               ####
#==============================================================================

resultats <- data.frame(
  Model = c("Numèric amb k=3", "Numèric amb k=1", "Mixt amb k=1", "Mixt amb k=3", "SMOTE mixt amb k=29"),
  F1 = c(f1_1, f1_2, f1_3, f1_4, f1_5),
  Accuracy = c(Accuracy_1, Accuracy_2, Accuracy_3, Accuracy_4, Accuracy_5),
  Sensitivity = c(Sensitivity_1, Sensitivity_2, Sensitivity_3, Sensitivity_4, Sensitivity_5),
  Specificity = c(Specificity_1, Specificity_2, Specificity_3, Specificity_4, Specificity_5),
  AUC = c(auc_value_1, auc_value_2, auc_value_3, auc_value_4, auc_value_5),
  Recall = c(recall_1, recall_2, recall_3, recall_4, recall_5)
)

print(resultats)

save(knn_mixto, knn_mixto_k3, best_model, best_model_1, best_model_mixto,
     best_model_smote, model_smote, resultats, file = "models_knn.RData")


