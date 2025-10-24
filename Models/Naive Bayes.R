library(caret)
library(naivebayes)
library(smotefamily)
library(MLmetrics)
library(dplyr)

set.seed(123)

data <- data_imputed_AREG[ , -17]

Index <- sample(1:nrow(data), size = nrow(data)*0.8)
dataTrain <- data[Index, ]
dataTest  <- data[-Index, ]

dataTrain$Exited <- factor(dataTrain$Exited, levels = c(0,1), labels = c("No", "Yes"))
dataTest$Exited  <- factor(dataTest$Exited,  levels = c(0,1), labels = c("No", "Yes"))

# F1
f1 <- function(data, lev = NULL, model = NULL) {
  f1_val <- F1_Score(y_pred = data$pred, y_true = data$obs, positive = "Yes")
  c(F1 = f1_val)
}

control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = f1,
  sampling = "smote",
  verboseIter = TRUE
)

# Grid inicial
grid <- expand.grid(
  usekernel = c(TRUE, FALSE),
  laplace = c(0, 0.5, 1),
  adjust = c(0, 0.5, 1, 1.5)
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
  laplace = 0,
  adjust = c(0,0.5,1,2,3,4,5)
)

modelo_nb2 <- train(
  Exited ~ .,
  data = dataTrain,
  method = "naive_bayes",
  trControl = control,
  tuneGrid = grid2,
  metric = "F1"
)

print(modelo_nb2)


# Predicciones (selecciona automaticamente los mejores parametros)
pred_class <- predict(modelo_nb2, newdata = dataTest, type = "raw")
pred_prob  <- predict(modelo_nb2, newdata = dataTest, type = "prob")
cm <- confusionMatrix(pred_class, dataTest$Exited, positive = "Yes")
print(cm)


# Esto es una mierda, se puede hacer tb tuning del llindar
probs <- predict(modelo_nb2, newdata = dataTest, type = "prob")[, "Yes"]
preds <- ifelse(probs > 0.2, "Yes", "No")
confusionMatrix(factor(preds, levels=c("No","Yes")), dataTest$Exited, positive="Yes")
