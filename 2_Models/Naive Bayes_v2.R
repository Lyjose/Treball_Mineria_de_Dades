
packages <- c("caret", "naivebayes", "smotefamily", "MLmetrics","dplyr")

install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

lapply(packages, install_if_missing)

set.seed(123)

load("dataAREG_outliers.RData")
data <- data_imputed_AREG[,-c(7,18)]

seleccio = c("Age","IsActiveMember","MaritalStatus","EstimatedSalary", 
             "SavingsAccountFlag","NumOfProducts","Gender","AvgTransactionAmount",
             "Geography","EducationLevel","HasCrCard")

Index <- sample(1:nrow(data), size = nrow(data)*0.8)
dataTrain <- data[Index, ]
dataTest  <- data[-Index, ]
# Filtrar las variables seleccionadas del conjunto de datos
dataTrain_subset <- dataTrain[, c("Exited", seleccio)]
dataTest_subset <- dataTrain[, c("Exited", seleccio)]

dataTrain$Exited <- factor(dataTrain$Exited, levels = c(0,1), labels = c("No", "Yes"))
dataTest$Exited  <- factor(dataTest$Exited,  levels = c(0,1), labels = c("No", "Yes"))

# F1
f1 <- function(data, lev = NULL, model = NULL) {
  f1_val <- MLmetrics::F1_Score(y_pred = data$pred, y_true = data$obs, positive = "Yes")
  c(F1 = f1_val)
}


control <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 10,
  classProbs = TRUE,
  summaryFunction = f1,
  sampling = "smote",
  verboseIter = TRUE
)

# Grid inicial
grid <- expand.grid(
  usekernel = c(TRUE, FALSE),
  laplace = c(0, 1, 2),
  adjust = c(0, 1, 2)
)

# Entrenamiento inicial
modelo_nb <- train(
  Exited ~ .,
  data = dataTrain,
  method = "naive_bayes",
  trControl = control,
  tuneGrid = grid,
  metric = "F1"
)

print(modelo_nb)

# Ajuste fino
grid2 <- expand.grid(
  usekernel = FALSE,
  laplace = c(0,1,2),
  adjust = 0
)

modelo_nb2 <- train(
  Exited ~ .,
  data = dataTrain,
  method = "naive_bayes",
  trControl = control,
  tuneGrid = grid,
  metric = "F1"
)

print(modelo_nb2)


# Predicciones (selecciona automaticamente los mejores parametros)
pred_class <- predict(modelo_nb2, newdata = dataTest, type = "raw")
pred_prob  <- predict(modelo_nb2, newdata = dataTest, type = "prob")
cm <- confusionMatrix(pred_class, dataTest$Exited, positive = "Yes")
print(cm)


################################################################################
#####                            THRESHOLD                                ####
################################################################################


# Esto es una mierda, se puede hacer tb tuning del llindar
probs <- predict(modelo_nb2, newdata = dataTest, type = "prob")[, "Yes"]
preds <- ifelse(probs > 0.2, "Yes", "No")
confusionMatrix(factor(preds, levels=c("No","Yes")), dataTest$Exited, positive="Yes")

thresholds <- seq(0.01, 0.8, by = 0.01)

# Inicializar un vector para almacenar los resultados del F1
f1_scores <- numeric(length(thresholds))

# Loop para probar cada threshold y calcular el F1
for (i in seq_along(thresholds)) {
  threshold <- thresholds[i]
  preds <- ifelse(probs > threshold, "Yes", "No")
  f1_scores[i] <- MLmetrics::F1_Score(y_pred = preds, y_true = dataTest$Exited, positive = "Yes")
}

threshold_f1_df <- data.frame(Threshold = thresholds, F1_Score = f1_scores)

print(threshold_f1_df)

best_threshold <- threshold_f1_df$Threshold[which.max(threshold_f1_df$F1_Score)]
best_threshold

preds_testtrain <- ifelse(probs > best_threshold, "Yes", "No")
# CM
cm_best <- confusionMatrix(
  factor(preds_testtrain, levels = c("No", "Yes")),
  dataTest$Exited,
  positive = "Yes"
); print(cm_best)

MLmetrics::F1_Score(dataTest$Exited,preds_testtrain,"Yes")

################################################################################
#####                            Para Kaggle                                ####
################################################################################

# 1. Obtener probabilidades de test
probs_test <- predict(modelo_nb2, newdata = data_imputed_AREG_test[-6], type = "prob")[, "Yes"]

# 2. Aplicar el mejor threshold encontrado
best_threshold <- threshold_f1_df$Threshold[which.max(threshold_f1_df$F1_Score)]
pred_test <- ifelse(probs_test > best_threshold, "Yes", "No")


# 3. Crear el dataframe de submission
submission <- data.frame(
  ID = data_imputed_AREG_test$ID,
  Exited = pred_test
)

# 4. Guardar el CSV
write.csv(submission, "submission_nb_outliers_Smote.csv", row.names = FALSE)
