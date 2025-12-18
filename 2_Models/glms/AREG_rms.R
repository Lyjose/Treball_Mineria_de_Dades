
packages <- c("caret", "naivebayes", "smotefamily", "MLmetrics","dplyr", "Hmisc","rms")

install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

lapply(packages, install_if_missing)

Mejores_thresholds <- c()

set.seed(sample(1:1000, 1))

load("dataAREG_outliers.RData")
data <- dataAREG

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

# Inicializamos los datos para el paquete rms.
#Se guardaran valores de las variables como media, moda y demás información descriptiva previa.

dd <- datadist(dataTrain)
dd

#Definimos dd como la base principal para rms
options(datadist = "dd")

#Debido a que AREG sirve muy bien para variables continuas pero no esta pensado para binarias
# debemos hacer un ajuste "logit" (lrm) para poder hacer las predicciones de Exited:

model_AREG <- glm(Exited ~ rcs(Tenure, 3) + rcs(NetPromoterScore, 3) +
                    rcs(TransactionFrequency, 3) + rcs(Age, 4) + rcs(EstimatedSalary, 4) + 
                    rcs(AvgTransactionAmount, 4) + rcs(DigitalEngagementScore, 4) + 
                    rcs(CreditScore, 4) + rcs(Balance, 4) + Gender + EducationLevel + 
                    Geography + HasCrCard + IsActiveMember + SavingsAccountFlag + 
                    NumOfProducts,
                  data = dataTrain,family=binomial(link="logit"))

# Notar que no usamos ni CustomerSegment ni ComplaintsCount ni MaritalStatus 
# No funcnionan correctamente, miraré de corregirlo.

summary(model_AREG)

# Prediccions de probabilitat
probs <- predict(model_AREG, newdata = dataTest, type = "response")  


# Convertir a prediccions binàries amb llindar 0.5
pred <- ifelse(probs > 0.5, "Yes", "No")

results <- data.frame(
  pred = pred,
  obs  = dataTest$Exited
)

f1_val <- f1(data = results)
f1_val



################################################################################
#####                            THRESHOLD                                ####
################################################################################

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

Mejores_thresholds <- c(Mejores_thresholds, best_threshold)

mean(Mejores_thresholds)


preds_testtrain <- ifelse(probs > best_threshold, "Yes", "No")
# CM
cm_best <- confusionMatrix(
  factor(preds_testtrain, levels = c("No", "Yes")),
  dataTest$Exited,
  positive = "Yes"
); print(cm_best)



################################################################################
#####                            Para Kaggle                                ####
################################################################################

# 1. Obtener probabilidades de test
probs_test <- predict(model_AREG, newdata = data_imputed_AREG_test[,-c(1,6)], type = "response")

# 2. Aplicar el mejor threshold encontrado
best_threshold <- threshold_f1_df$Threshold[which.max(threshold_f1_df$F1_Score)]
pred_test <- ifelse(probs_test > mean(Mejores_thresholds), "Yes", "No")


# 3. Crear el dataframe de submission
submission <- data.frame(
  ID = data_imputed_AREG_test$ID,
  Exited = pred_test
)

# 4. Guardar el CSV
write.csv(submission, "submission_AREG.csv", row.names = FALSE)
