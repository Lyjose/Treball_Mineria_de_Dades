
packages <- c("caret", "naivebayes", "smotefamily", "MLmetrics","dplyr", "Hmisc","rms")

install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

lapply(packages, install_if_missing)

set.seed(123)

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

model_AREG <- lrm(Exited ~ rcs(Tenure, 3) + rcs(NetPromoterScore, 3) +
                    rcs(TransactionFrequency, 3) + rcs(Age, 4) + rcs(EstimatedSalary, 4) + 
                    rcs(AvgTransactionAmount, 4) + rcs(DigitalEngagementScore, 4) + 
                    rcs(CreditScore, 4) + rcs(Balance, 4) + Gender + EducationLevel + 
                    LoanStatus + Geography + HasCrCard + IsActiveMember +
                    SavingsAccountFlag + NumOfProducts,  
                  data = dataTrain)

# Notar que no usamos ni CustomerSegment ni ComplaintsCount ni MaritalStatus 
# No funcnionan correctamente, miraré de corregirlo.


# Prediccions de probabilitat
probs <- predict(model_AREG, newdata = dataTest, type = "fitted")  

# Convertir a prediccions binàries amb llindar 0.5
pred <- ifelse(probs > 0.8, "Yes", "No")

results <- data.frame(
  pred = pred,
  obs  = dataTest$Exited
)

f1_val <- f1(data = results)
f1_val













################################################################################
#####                            Para Kaggle                                ####
################################################################################

# 1. Obtener probabilidades de test
probs_test <- predict(modelo_nb2, newdata = data_imputed_AREG_test, type = "prob")[, "Yes"]

# 2. Aplicar el mejor threshold encontrado
best_threshold <- threshold_f1_df$Threshold[which.max(threshold_f1_df$F1_Score)]
pred_test <- ifelse(probs_test > best_threshold, "Yes", "No")


# 3. Crear el dataframe de submission
submission <- data.frame(
  ID = data_imputed_AREG_test$ID,
  Exited = pred_test
)

# 4. Guardar el CSV
write.csv(submission, "submission_nb_smote.csv", row.names = FALSE)
