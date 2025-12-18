library(caret)
library(dplyr)
library(MLmetrics)

set.seed(123)

# ==============================================================================
# 1. FUNCIÓN PARA CREAR DUMMIES (Con R Base)
# ==============================================================================
convert_to_dummies <- function(df) {
  # model.matrix es nativo de R. 
  # "~ Geography + Gender - 1" significa: crea dummies para estas vars y quita el intercepto.
  # El "-1" es importante para tener todas las categorías (one-hot) o evitar multicolinealidad perfecta
  # dependiendo de si luego usas intercepto en el modelo, pero SVM lo maneja bien.
  
  dummies <- model.matrix(~ Geography + Gender - 1, data = df)
  
  # Convertimos a dataframe
  df_dummies <- as.data.frame(dummies)
  
  # Convertimos HasCrCard e IsActiveMember a numérico (0 y 1) 
  # (R a veces los lee como factores 1 y 2, esto lo corrige a 0 y 1 real)
  if(is.factor(df$HasCrCard)) {
    df$HasCrCard <- as.numeric(as.character(df$HasCrCard))
  }
  if(is.factor(df$IsActiveMember)) {
    df$IsActiveMember <- as.numeric(as.character(df$IsActiveMember))
  }
  
  # Unimos las dummies y quitamos las variables categóricas originales
  df_final <- cbind(df, df_dummies)
  df_final$Geography <- NULL
  df_final$Gender <- NULL
  
  return(df_final)
}

# ==============================================================================
# 2. PROCESO DE TRANSFORMACIÓN
# ==============================================================================

# A. Polinomios (Tal cual lo tenías)
data$Age2 <- data$Age^2
data$Age3 <- data$Age^3
data$Age4 <- data$Age^4

dataAREG_test_final$Age2 <- dataAREG_test_final$Age^2
dataAREG_test_final$Age3 <- dataAREG_test_final$Age^3
dataAREG_test_final$Age4 <- dataAREG_test_final$Age^4

# B. Aplicar Dummies
data_proc <- convert_to_dummies(data)
data_kaggle_proc <- convert_to_dummies(dataAREG_test_final)

# Asegurar mismas columnas (intersección) por si acaso
common_cols <- setdiff(names(data_proc), "Exited") # Exited solo está en train
data_kaggle_proc <- data_kaggle_proc[, common_cols]

# ==============================================================================
# 3. SPLIT Y ESCALADO MANUAL
# ==============================================================================

Index <- createDataPartition(data_proc$Exited, p = 0.8, list = FALSE)
dataTrain <- data_proc[Index, ]
dataTest  <- data_proc[-Index, ]

vars_to_scale <- c("CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", 
                   "EstimatedSalary", "Age2", "Age3", "Age4")

# Calcular medias y desviaciones SOLO en Train
train_means <- sapply(dataTrain[, vars_to_scale], mean)
train_sds   <- sapply(dataTrain[, vars_to_scale], sd)

# Función simple de escalado
manual_scale <- function(df, vars, means, sds) {
  for (v in vars) {
    df[[v]] <- (df[[v]] - means[v]) / sds[v]
  }
  return(df)
}

# Aplicar escalado
dataTrain <- manual_scale(dataTrain, vars_to_scale, train_means, train_sds)
dataTest  <- manual_scale(dataTest, vars_to_scale, train_means, train_sds)
data_kaggle_proc <- manual_scale(data_kaggle_proc, vars_to_scale, train_means, train_sds)

# ==============================================================================
# 5. PREPARACIÓN PARA CARET (Target y F1)
# ==============================================================================
# Caret necesita factores válidos para nombres de variables (No 0/1, sino Yes/No)
dataTrain$Exited <- factor(dataTrain$Exited, levels = c(0, 1), labels = c("No", "Yes"))
dataTest$Exited  <- factor(dataTest$Exited, levels = c(0, 1), labels = c("No", "Yes"))

# Tu función F1 personalizada
f1_custom <- function(data, lev = NULL, model = NULL) {
  f1_val <- MLmetrics::F1_Score(y_pred = data$pred, y_true = data$obs, positive = "Yes")
  c(F1 = f1_val)
}

ctrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5, # Bajado a 5 por velocidad, sube a 10 si tienes tiempo
  classProbs = TRUE,
  summaryFunction = f1_custom, # Usamos tu función para maximizar F1 en el tuning
  verboseIter = TRUE,
  savePredictions = "final"
)

# ==============================================================================
# 6. ENTRENAMIENTO DEL MODELO (SVM Radial)
# ==============================================================================
# Grid de hiperparámetros (puedes ampliarlo)
grid_svm <- expand.grid(
  sigma = c(0.01, 0.05, 0.1),
  C = c(0.1, 1, 10)
)

print("Entrenando SVM...")
model_svm <- train(
  Exited ~ ., 
  data = dataTrain,
  method = "svmRadial",
  trControl = ctrl,
  metric = "F1",
  tuneGrid = data.frame(C=0.1,sigma=0.01)
)

print(model_svm)

# ==============================================================================
# 7. OPTIMIZACIÓN DEL THRESHOLD (Umbral de decisión)
# ==============================================================================

# 3. Optimitzem el Threshold usant les prediccions del CROSS-VALIDATION
# Accedim a les prediccions internes del millor model
cv_preds <- model_svm$pred

# Busquem el millor tall sobre aquestes prediccions 'netes'
thresholds <- seq(0.1, 0.9, by = 0.01)
f1_scores_cv <- numeric(length(thresholds))

for (i in seq_along(thresholds)) {
  thresh <- thresholds[i]
  # Compte: a model_svm$pred, la columna de probabilitat sol dir-se "Yes"
  preds_class <- factor(ifelse(cv_preds$Yes > thresh, "Yes", "No"), levels = c("No", "Yes"))
  # I la columna real es diu "obs"
  f1_scores_cv[i] <- MLmetrics::F1_Score(y_true = cv_preds$obs, y_pred = preds_class, positive = "Yes")
}

best_thresh <- thresholds[which.max(f1_scores_cv)]
cat("Millor Threshold (estimat per CV):", best_thresh, "\n")

# 4. AVALUACIÓ FINAL (Ara sí, toquem el Test per veure la realitat)
# Aquest pas només és per saber quina nota treuries, NO per decidir res.
probs_test <- predict(model_svm, newdata = dataTest, type = "prob")$Yes
pred_test_class <- factor(ifelse(probs_test > best_thresh, "Yes", "No"), levels = c("No", "Yes"))

f1_final_test <- MLmetrics::F1_Score(y_true = dataTest$Exited, y_pred = pred_test_class, positive = "Yes")
cat("F1 Real al Test Set (sense fer trampes):", f1_final_test, "\n")
# ==============================================================================
# 8. GENERAR SUBMISSION (Kaggle)
# ==============================================================================

# 1. Predecir probabilidades sobre el set final arreglado
probs_kaggle <- predict(model_svm, newdata = data_kaggle_proc, type = "prob")$Yes

# 2. Aplicar el mejor threshold encontrado
pred_kaggle_class <- ifelse(probs_kaggle > best_thresh, "Yes", "No") # Kaggle suele pedir 0/1

# 3. Crear dataframe
submission <- data.frame(
  ID = dataAREG_test_final$ID, # Recuperamos el ID del original
  Exited = pred_kaggle_class
)

head(submission)
write.csv(submission, "submission_svm_optimized.csv", row.names = FALSE)
