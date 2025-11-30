###############################################################################
# 0. BLOQUE ÚNICO: Paquetes, utilidades y preparación global
###############################################################################
packages <- c(
  "caret", "naivebayes", "smotefamily", "MLmetrics", "dplyr",
  "glmnet", "lightgbm", "Matrix", "data.table", "e1071", "splines"
)
library(dplyr)
install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
    library(pkg, character.only = TRUE)
  }
}
invisible(lapply(packages, install_if_missing))

set.seed(123)

# --- Asumo que ya tienes en el environment:
# data_imputed_AREG  (train+val dataset con target Exited como factor 0/1)
# data_imputed_AREG_test (test para submission; contiene ID)
# Si no, carga tus datos aquí.

###############################################################################
# 1) PAQUETES, DIVISIÓN DATOS, FUNCIÓN F1, CONTROL (REPEATEDCV 10x10),
#    preparación variable respuesta como factor, train/test
###############################################################################

# --- Eliminamos Surname e ID del dataset de training
data <- data_imputed_AREG %>% dplyr::select(-Surname, -ID)

# Convertimos Exited a factor "No"/"Yes" si no lo está en ese formato
if(!("Exited" %in% names(data))) stop("No encuentro la variable 'Exited' en data")
data$Exited <- factor(as.character(data$Exited), levels = c("0","1"), labels = c("No","Yes"))

# Split 80/20
set.seed(123)
Index <- sample(1:nrow(data), size = floor(nrow(data)*0.8))
dataTrain <- data[Index, ]
dataTest  <- data[-Index, ]

# F1 function
f1_summary <- function(data, lev = NULL, model = NULL) {
  f1_val <- MLmetrics::F1_Score(y_pred = data$pred, y_true = data$obs, positive = "Yes")
  c(F1 = f1_val)
}

control <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 10,
  classProbs = TRUE,
  summaryFunction = f1_summary,
  verboseIter = TRUE,
  savePredictions = "final"
)

# Aseguramos que test también tenga niveles correctos
dataTest$Exited <- factor(as.character(dataTest$Exited), levels = c("No","Yes"))

###############################################################################
# 1.5) FEATURE ENGINEERING: Splines, Interacciones, Ratios, Flags
###############################################################################
cat("=== INICIANDO FEATURE ENGINEERING ===\n")

# Función para aplicar todas las transformaciones a un dataset
apply_feature_engineering <- function(df, is_train = TRUE) {
  
  # Guardamos el target si existe
  has_target <- "Exited" %in% names(df)
  if(has_target) target <- df$Exited
  
  # ------------------------------------------------------------------
  # A) SPLINES en variables numéricas clave (4 df en Age, 3 en otras)
  # ------------------------------------------------------------------
  library(splines)
  
  # Age con 4 grados de libertad
  age_splines <- ns(df$Age, df = 4)
  colnames(age_splines) <- paste0("Age_spline_", 1:4)
  
  # Balance con 3 df (con manejo de warnings por valores idénticos)
  balance_splines <- tryCatch({
    ns(df$Balance, df = 3)
  }, warning = function(w) {
    # Si hay warning por knots duplicados, usar menos df
    ns(df$Balance, df = 2)
  })
  # Ajustar nombres de columnas según df real
  colnames(balance_splines) <- paste0("Balance_spline_", 1:ncol(balance_splines))
  
  # EstimatedSalary con 3 df
  salary_splines <- ns(df$EstimatedSalary, df = 3)
  colnames(salary_splines) <- paste0("Salary_spline_", 1:3)
  
  # CreditScore con 3 df
  credit_splines <- ns(df$CreditScore, df = 3)
  colnames(credit_splines) <- paste0("Credit_spline_", 1:3)
  
  # ------------------------------------------------------------------
  # B) RATIOS FINANCIEROS (métricas clave de churn bancario)
  # ------------------------------------------------------------------
  df$Balance_per_Salary <- ifelse(df$EstimatedSalary > 0, 
                                  df$Balance / df$EstimatedSalary, 0)
  
  df$Balance_per_Tenure <- ifelse(df$Tenure > 0, 
                                  df$Balance / df$Tenure, 0)
  
  df$Salary_per_Tenure <- ifelse(df$Tenure > 0,
                                 df$EstimatedSalary / df$Tenure, 0)
  
  df$AvgTransaction_per_Frequency <- ifelse(df$TransactionFrequency > 0,
                                            df$AvgTransactionAmount / df$TransactionFrequency, 0)
  
  df$Balance_per_Product <- df$Balance / as.numeric(as.character(df$NumOfProducts))
  
  df$CreditScore_per_Age <- df$CreditScore / df$Age
  
  # ------------------------------------------------------------------
  # C) FLAGS DE RIESGO (indicadores binarios de churn)
  # ------------------------------------------------------------------
  df$Flag_ZeroBalance <- factor(ifelse(df$Balance == 0, 1, 0))
  
  df$Flag_SingleProduct <- factor(ifelse(df$NumOfProducts == "1", 1, 0))
  
  df$Flag_Inactive <- factor(ifelse(df$IsActiveMember == "0", 1, 0))
  
  df$Flag_NoCreditCard <- factor(ifelse(df$HasCrCard == "0", 1, 0))
  
  df$Flag_LowEngagement <- factor(ifelse(df$DigitalEngagementScore < 50, 1, 0))
  
  df$Flag_LowNPS <- factor(ifelse(df$NetPromoterScore <= 6, 1, 0))
  
  df$Flag_HighComplains <- factor(ifelse(as.numeric(as.character(df$ComplaintsCount)) >= 2, 1, 0))
  
  df$Flag_SeniorAge <- factor(ifelse(df$Age >= 60, 1, 0))
  
  df$Flag_ActiveLoan <- factor(ifelse(df$LoanStatus == "Active loan", 1, 0))
  
  # Combinación de flags de riesgo (suma de factores de riesgo)
  df$RiskScore <- as.numeric(df$Flag_ZeroBalance == "1") +
    as.numeric(df$Flag_SingleProduct == "1") +
    as.numeric(df$Flag_Inactive == "1") +
    as.numeric(df$Flag_LowEngagement == "1") +
    as.numeric(df$Flag_LowNPS == "1") +
    as.numeric(df$Flag_HighComplains == "1")
  
  # ------------------------------------------------------------------
  # D) INTERACCIONES DE DOMINIO BANCARIO (las más relevantes para churn)
  # ------------------------------------------------------------------
  # Age * Balance (clientes mayores con alto balance son más estables)
  df$Age_x_Balance <- df$Age * df$Balance
  
  # Age * Tenure (antigüedad ajustada por edad)
  df$Age_x_Tenure <- df$Age * df$Tenure
  
  # NumOfProducts * Balance (diversificación de productos)
  df$NumProducts_x_Balance <- as.numeric(as.character(df$NumOfProducts)) * df$Balance
  
  # IsActiveMember * Balance (engagement financiero)
  df$Active_x_Balance <- as.numeric(df$IsActiveMember == "1") * df$Balance
  
  # Geography * Balance (interacción geográfica con balance)
  df$Germany_x_Balance <- as.numeric(df$Geography == "Germany") * df$Balance
  df$France_x_Balance <- as.numeric(df$Geography == "France") * df$Balance
  
  # CreditScore * Age
  df$Credit_x_Age <- df$CreditScore * df$Age
  
  # NumOfProducts * Tenure (lealtad por productos)
  df$NumProducts_x_Tenure <- as.numeric(as.character(df$NumOfProducts)) * df$Tenure
  
  # TransactionFrequency * AvgTransactionAmount (volumen total transaccional)
  df$TotalTransactionVolume <- df$TransactionFrequency * df$AvgTransactionAmount
  
  # Engagement * NPS (satisfacción digital)
  df$Engagement_x_NPS <- df$DigitalEngagementScore * df$NetPromoterScore
  
  # ------------------------------------------------------------------
  # E) SEGMENTACIONES (categorización de variables continuas)
  # ------------------------------------------------------------------
  # Edad en grupos (con manejo robusto)
  df$Age_Group <- cut(df$Age, 
                      breaks = c(0, 30, 40, 50, 60, 100),
                      labels = c("Young", "Adult", "MidAge", "Senior", "Elder"),
                      include.lowest = TRUE,
                      right = FALSE)  # evita problemas de límites
  
  # Balance en quintiles (con manejo de valores duplicados)
  balance_breaks <- unique(quantile(df$Balance, probs = seq(0, 1, 0.2), na.rm = TRUE))
  if(length(balance_breaks) < 2) {
    # Si todos los balances son iguales, crear segmento único
    df$Balance_Segment <- factor(rep("Medium", nrow(df)))
  } else if(length(balance_breaks) < 6) {
    # Si hay menos de 6 breaks únicos, usar los que hay
    n_labels <- length(balance_breaks) - 1
    labels_available <- c("VeryLow", "Low", "Medium", "High", "VeryHigh")[1:n_labels]
    df$Balance_Segment <- cut(df$Balance,
                              breaks = balance_breaks,
                              labels = labels_available,
                              include.lowest = TRUE)
  } else {
    df$Balance_Segment <- cut(df$Balance,
                              breaks = balance_breaks,
                              labels = c("VeryLow", "Low", "Medium", "High", "VeryHigh"),
                              include.lowest = TRUE)
  }
  
  # Tenure en grupos (con manejo robusto)
  df$Tenure_Group <- cut(df$Tenure,
                         breaks = c(-0.1, 2, 5, 8, 15),
                         labels = c("VeryNew", "New", "Established", "Loyal"),
                         include.lowest = TRUE,
                         right = TRUE)
  
  # CreditScore en categorías (con manejo robusto)
  df$Credit_Category <- cut(df$CreditScore,
                            breaks = c(0, 500, 650, 750, 1000),
                            labels = c("Poor", "Fair", "Good", "Excellent"),
                            include.lowest = TRUE,
                            right = FALSE)
  
  # ------------------------------------------------------------------
  # F) AGREGACIONES Y MÉTRICAS COMPUESTAS
  # ------------------------------------------------------------------
  # Índice de valor del cliente (CLV proxy)
  df$Customer_Value_Index <- (df$Balance * 0.4 + 
                                df$EstimatedSalary * 0.3 + 
                                as.numeric(as.character(df$NumOfProducts)) * 20000 * 0.3) / 100000
  
  # Score de compromiso total
  df$Engagement_Total <- (df$DigitalEngagementScore * 0.5 + 
                            df$NetPromoterScore * 5 * 0.3 +
                            as.numeric(df$IsActiveMember == "1") * 50 * 0.2)
  
  # Ratio de productos por antigüedad
  df$Products_per_Tenure <- as.numeric(as.character(df$NumOfProducts)) / pmax(df$Tenure, 1)
  
  # ------------------------------------------------------------------
  # G) AÑADIR SPLINES AL DATAFRAME
  # ------------------------------------------------------------------
  df <- cbind(df, age_splines, balance_splines, salary_splines, credit_splines)
  
  # Restaurar target si existía
  if(has_target) {
    df$Exited <- target
  }
  
  return(df)
}

# Aplicar FE a train, test y kaggle
cat("Aplicando FE a dataTrain...\n")
dataTrain <- apply_feature_engineering(dataTrain, is_train = TRUE)

cat("Aplicando FE a dataTest...\n")
dataTest <- apply_feature_engineering(dataTest, is_train = FALSE)

cat("Aplicando FE a kaggle dataset...\n")
# Preparar kaggle dataset (eliminar ID primero, lo guardamos aparte)
kaggle_raw <- data_imputed_AREG_test %>% dplyr::select(-Surname)
kaggle_ids <- kaggle_raw$ID
kaggle_raw <- kaggle_raw %>% dplyr::select(-ID)
kaggle_raw <- apply_feature_engineering(kaggle_raw, is_train = FALSE)

cat("=== FEATURE ENGINEERING COMPLETADO ===\n")
cat("Dimensiones dataTrain:", dim(dataTrain), "\n")
cat("Dimensiones dataTest:", dim(dataTest), "\n")
cat("Dimensiones kaggle_raw:", dim(kaggle_raw), "\n")

###############################################################################
# UTIL: preparar matrices/dummies para glmnet / lightgbm (usar mismo diseño)
###############################################################################
# Creamos dummyVars sobre training (sin target)
dv <- caret::dummyVars(~ ., data = dataTrain %>% dplyr::select(-Exited), 
                       fullRank = TRUE)

# Aplico a train, test y kaggle
x_train_mat <- predict(dv, newdata = dataTrain %>% dplyr::select(-Exited)) %>% as.matrix()
x_test_mat  <- predict(dv, newdata = dataTest %>% dplyr::select(-Exited)) %>% as.matrix()
kaggle_x_mat <- predict(dv, newdata = kaggle_raw) %>% as.matrix()

# Vector de labels binarios para glmnet/lightgbm
y_train_num <- ifelse(dataTrain$Exited == "Yes", 1L, 0L)
y_test_num  <- ifelse(dataTest$Exited  == "Yes", 1L, 0L)

# Calculo de pesos/ratio para usar en LASSO y LightGBM
n_pos <- sum(y_train_num == 1)
n_neg <- sum(y_train_num == 0)
scale_pos_weight <- ifelse(n_pos == 0, 1, n_neg / n_pos)
class_weights <- c("0" = 1, "1" = n_neg / n_pos)

###############################################################################
# 2) MODELO: Logistic LASSO (cv.glmnet) - con weights
###############################################################################
library(glmnet)
x_train_sparse <- Matrix::Matrix(x_train_mat, sparse = TRUE)

# Hacemos cv.glmnet con family binomial y weights (balanceo)
cv_lasso <- cv.glmnet(
  x = x_train_sparse,
  y = y_train_num,
  family = "binomial",
  alpha = 1,
  nfolds = 5,
  type.measure = "class",
  weights = ifelse(y_train_num == 1, class_weights["1"], class_weights["0"])
)
best_lambda <- cv_lasso$lambda.min
cat("LASSO - Best lambda:", best_lambda, "\n")

# Probabilidades del lasso sobre el conjunto test
x_test_sparse <- Matrix::Matrix(x_test_mat, sparse = TRUE)
lasso_prob_test <- predict(cv_lasso, newx = x_test_sparse, s = "lambda.min", type = "response")[,1]

###############################################################################
# 3) MODELO: Naive Bayes (usando pipeline con caret y sampling = "smote")
###############################################################################
grid_nb <- expand.grid(
  usekernel = c(TRUE, FALSE),
  laplace = c(0, 1, 2),
  adjust = c(0, 1, 2)
)

set.seed(123)
modelo_nb_final <- caret::train(
  Exited ~ .,
  data = dataTrain,
  method = "naive_bayes",
  trControl = control,
  tuneGrid = grid_nb,
  metric = "F1"
)

print(modelo_nb_final)

# Predicciones probabilísticas NB sobre test
nb_prob_test <- predict(modelo_nb_final, newdata = dataTest, type = "prob")[, "Yes"]

###############################################################################
# 4) MODELO: LightGBM (entrenamiento final sobre todo dataTrain)
###############################################################################
library(lightgbm)
dtrain_lgb <- lgb.Dataset(data = x_train_mat, label = y_train_num)

params_lgb <- list(
  objective = "binary",
  metric = "binary_logloss",
  boosting = "gbdt",
  learning_rate = 0.05,
  num_leaves = 31,
  feature_fraction = 0.9,
  bagging_fraction = 0.8,
  bagging_freq = 5,
  min_data_in_leaf = 20,
  verbose = -1,
  scale_pos_weight = scale_pos_weight
)

set.seed(123)
lgb_model_final <- lgb.train(
  params = params_lgb,
  data = dtrain_lgb,
  nrounds = 1000,
  valids = list(train = dtrain_lgb),
  early_stopping_rounds = 50,
  verbose = -1
)

# Predicciones probabilísticas LightGBM sobre test
lgb_prob_test <- predict(lgb_model_final, x_test_mat)

###############################################################################
# 5) METAMODELO: construimos predicciones OOF para entrenar el meta (5-fold)
###############################################################################
library(caret)
folds <- createFolds(dataTrain$Exited, k = 5, list = TRUE, returnTrain = FALSE)

# Reservas para OOF
oof_nb   <- numeric(nrow(dataTrain))
oof_lasso <- numeric(nrow(dataTrain))
oof_lgb  <- numeric(nrow(dataTrain))

for(i in seq_along(folds)) {
  cat("Fold", i, "\n")
  val_idx <- folds[[i]]
  train_idx <- setdiff(seq_len(nrow(dataTrain)), val_idx)
  
  # Subsets
  dtr <- dataTrain[train_idx, ]
  dval <- dataTrain[val_idx, ]
  
  # --- NB
  nb_fit <- naivebayes::naive_bayes(Exited ~ ., data = dtr)
  nb_preds <- predict(nb_fit, newdata = dval, type = "prob")[, "Yes"]
  oof_nb[val_idx] <- nb_preds
  
  # --- LASSO
  x_dtr <- predict(dv, newdata = dtr %>% dplyr::select(-Exited)) %>% as.matrix()
  x_dval <- predict(dv, newdata = dval %>% dplyr::select(-Exited)) %>% as.matrix()
  y_dtr <- ifelse(dtr$Exited == "Yes", 1L, 0L)
  
  cv_tmp <- cv.glmnet(
    x = Matrix::Matrix(x_dtr, sparse = TRUE),
    y = y_dtr,
    family = "binomial",
    alpha = 1,
    nfolds = 5,
    type.measure = "class",
    weights = ifelse(y_dtr == 1, class_weights["1"], class_weights["0"])
  )
  lasso_fold_prob <- predict(cv_tmp, newx = Matrix::Matrix(x_dval, sparse = TRUE), s = "lambda.min", type = "response")[,1]
  oof_lasso[val_idx] <- lasso_fold_prob
  
  # --- LightGBM
  dtr_mat <- predict(dv, newdata = dtr %>% dplyr::select(-Exited)) %>% as.matrix()
  dval_mat <- predict(dv, newdata = dval %>% dplyr::select(-Exited)) %>% as.matrix()
  y_dtr <- ifelse(dtr$Exited == "Yes", 1L, 0L)
  dtrain_tmp <- lgb.Dataset(data = dtr_mat, label = y_dtr)
  set.seed(123 + i)
  lgb_tmp <- lgb.train(
    params = params_lgb,
    data = dtrain_tmp,
    nrounds = 1000,
    valids = list(train = dtrain_tmp),
    early_stopping_rounds = 50,
    verbose = -1
  )
  lgb_fold_prob <- predict(lgb_tmp, dval_mat)
  oof_lgb[val_idx] <- lgb_fold_prob
}

# Comprobamos que no haya NA
stopifnot(!any(is.na(oof_nb)))
stopifnot(!any(is.na(oof_lasso)))
stopifnot(!any(is.na(oof_lgb)))

# Tabla OOF para meta-modelo
oof_df <- data.frame(
  nb_prob = oof_nb,
  lasso_prob = oof_lasso,
  lgb_prob = oof_lgb,
  target = dataTrain$Exited
)

# Entrenamos el meta-modelo (logistic) sobre OOF preds
meta_glm <- glm(I(target == "Yes") ~ nb_prob + lasso_prob + lgb_prob,
                data = oof_df, family = "binomial")

summary(meta_glm)

###############################################################################
# 6) TUNING THRESHOLD METAMODELO (y también optimizamos threshold para cada base)
###############################################################################
nb_prob_test_full   <- predict(modelo_nb_final, newdata = dataTest, type = "prob")[, "Yes"]
lasso_prob_test_full <- predict(cv_lasso, newx = x_test_sparse, s = "lambda.min", type = "response")[,1]
lgb_prob_test_full  <- predict(lgb_model_final, x_test_mat)

meta_input_test <- data.frame(
  nb_prob = nb_prob_test_full,
  lasso_prob = lasso_prob_test_full,
  lgb_prob = lgb_prob_test_full
)

meta_prob_test <- predict(meta_glm, newdata = meta_input_test, type = "response")

# Funcion para buscar mejor threshold maximizando F1
find_best_threshold <- function(probs, truth_factor, positives = "Yes", thr_seq = seq(0.01, 0.9, by = 0.01)) {
  f1s <- sapply(thr_seq, function(t) {
    preds <- factor(ifelse(probs > t, positives, ifelse(positives == "Yes", "No", "Yes")),
                    levels = levels(truth_factor))
    MLmetrics::F1_Score(y_pred = preds, y_true = truth_factor, positive = positives)
  })
  best_idx <- which.max(f1s)
  list(best_threshold = thr_seq[best_idx], best_f1 = f1s[best_idx], f1_by_thr = data.frame(thr = thr_seq, f1 = f1s))
}

# Optimizo thresholds
thr_nb   <- find_best_threshold(nb_prob_test_full, dataTest$Exited)
thr_lasso <- find_best_threshold(lasso_prob_test_full, dataTest$Exited)
thr_lgb  <- find_best_threshold(lgb_prob_test_full, dataTest$Exited)
thr_meta <- find_best_threshold(meta_prob_test, dataTest$Exited)

cat("NB threshold:", thr_nb$best_threshold, "F1:", thr_nb$best_f1, "\n")
cat("LASSO threshold:", thr_lasso$best_threshold, "F1:", thr_lasso$best_f1, "\n")
cat("LGBM threshold:", thr_lgb$best_threshold, "F1:", thr_lgb$best_f1, "\n")
cat("META threshold:", thr_meta$best_threshold, "F1:", thr_meta$best_f1, "\n")

###############################################################################
# 7) RESULTADOS: métricas y confusion matrices
###############################################################################
eval_model <- function(probs, thr, truth_factor) {
  preds <- factor(ifelse(probs > thr, "Yes", "No"), levels = levels(truth_factor))
  cm <- caret::confusionMatrix(preds, truth_factor, positive = "Yes")
  f1 <- MLmetrics::F1_Score(y_pred = preds, y_true = truth_factor, positive = "Yes")
  list(confusion = cm, F1 = f1)
}

res_nb    <- eval_model(nb_prob_test_full, thr_nb$best_threshold, dataTest$Exited)
res_lasso <- eval_model(lasso_prob_test_full, thr_lasso$best_threshold, dataTest$Exited)
res_lgb   <- eval_model(lgb_prob_test_full, thr_lgb$best_threshold, dataTest$Exited)
res_meta  <- eval_model(meta_prob_test, thr_meta$best_threshold, dataTest$Exited)

cat("\n=== RESULTADOS (usando umbral óptimo en test) ===\n")
cat("NB:   best_thr=", thr_nb$best_threshold, " F1=", res_nb$F1, "\n")
print(res_nb$confusion)
cat("\nLASSO: best_thr=", thr_lasso$best_threshold, " F1=", res_lasso$F1, "\n")
print(res_lasso$confusion)
cat("\nLGBM: best_thr=", thr_lgb$best_threshold, " F1=", res_lgb$F1, "\n")
print(res_lgb$confusion)
cat("\nMETA: best_thr=", thr_meta$best_threshold, " F1=", res_meta$F1, "\n")
print(res_meta$confusion)

###############################################################################
# 8) SUBMISSION PARA KAGGLE (usamos el meta-modelo + umbral óptimo del meta)
###############################################################################
# NB
nb_prob_kaggle <- predict(modelo_nb_final, newdata = kaggle_raw, type = "prob")[, "Yes"]

# LASSO y LGBM - alinear columnas
common_cols <- colnames(x_train_mat)
if(!all(common_cols %in% colnames(kaggle_x_mat))) {
  missing_cols <- setdiff(common_cols, colnames(kaggle_x_mat))
  if(length(missing_cols)) {
    add_mat <- matrix(0, nrow = nrow(kaggle_x_mat), ncol = length(missing_cols),
                      dimnames = list(NULL, missing_cols))
    kaggle_x_mat <- cbind(kaggle_x_mat, add_mat)
  }
}
kaggle_x_mat <- kaggle_x_mat[, common_cols, drop = FALSE]

lasso_prob_kaggle <- predict(cv_lasso, newx = Matrix::Matrix(kaggle_x_mat, sparse = TRUE),
                             s = "lambda.min", type = "response")[,1]
lgb_prob_kaggle <- predict(lgb_model_final, kaggle_x_mat)

meta_input_kaggle <- data.frame(
  nb_prob = nb_prob_kaggle,
  lasso_prob = lasso_prob_kaggle,
  lgb_prob = lgb_prob_kaggle
)

meta_prob_kaggle <- predict(meta_glm, newdata = meta_input_kaggle, type = "response")

# Aplicar el umbral óptimo del meta
best_meta_thr <- thr_meta$best_threshold
pred_kaggle <- ifelse(meta_prob_kaggle > best_meta_thr, "Yes", "No")

submission <- data.frame(
  ID = kaggle_ids,
  Exited = pred_kaggle
)

# Guardar CSV
write.csv(submission, file = "submission_meta_model_FE.csv", row.names = FALSE)
cat("\nSubmission saved to submission_meta_model_FE.csv (threshold meta:", best_meta_thr, ")\n")
cat("Features totales creadas:", ncol(x_train_mat), "\n")
