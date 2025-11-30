###############################################################################
# 0. BLOQUE ÚNICO: Paquetes, utilidades y preparación global
###############################################################################
packages <- c(
  "caret", "naivebayes", "smotefamily", "MLmetrics", "dplyr",
  "glmnet", "lightgbm", "Matrix", "data.table", "e1071"
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

# --- Partimos de tu código: hacemos sample 80/20 igual que antes
data <- data_imputed_AREG[ , -c(7,18) ]   # conserva tu operación original si quieres

# Convertimos Exited a factor "No"/"Yes" si no lo está en ese formato
if(!("Exited" %in% names(data))) stop("No encuentro la variable 'Exited' en data")
# Si Exited es factor con niveles "0","1", lo volvemos a etiquetar
data$Exited <- factor(as.character(data$Exited), levels = c("0","1"), labels = c("No","Yes"))

# Split 80/20 (igual que tu NB original)
set.seed(123)
Index <- sample(1:nrow(data), size = floor(nrow(data)*0.8))
dataTrain <- data[Index, ]
dataTest  <- data[-Index, ]

# F1 function (caret summaryFunction expects this signature)
f1_summary <- function(data, lev = NULL, model = NULL) {
  # data$pred are predicted labels, data$obs are true labels
  f1_val <- MLmetrics::F1_Score(y_pred = data$pred, y_true = data$obs, positive = "Yes")
  c(F1 = f1_val)
}

control <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 10,
  classProbs = TRUE,
  summaryFunction = f1_summary,
  sampling = "smote",         # se usará para entrenar via caret (NB final)
  verboseIter = TRUE,
  savePredictions = "final"   # útil si usamos caret para modelos
)

# Aseguramos que test también tenga niveles correctos
dataTest$Exited <- factor(as.character(dataTest$Exited), levels = c("No","Yes"))

###############################################################################
# UTIL: preparar matrices/dummies para glmnet / lightgbm (usar mismo diseño)
###############################################################################
# Creamos dummyVars sobre training (sin target)
dv <- caret::dummyVars(~ ., data = dataTrain %>% dplyr::select(-Exited), 
                       fullRank = TRUE)   # fullRank para evitar multicolinealidad perfecta

# Aplico a train, test y al dataset de kaggle (data_imputed_AREG_test)
x_train_mat <- predict(dv, newdata = dataTrain %>% dplyr::select(-Exited)) %>% as.matrix()
x_test_mat  <- predict(dv, newdata = dataTest %>% dplyr::select(-Exited)) %>% as.matrix()

# Para la tabla de submission (asumiendo que data_imputed_AREG_test tiene SAME columns except Exited)
kaggle_raw <- data_imputed_AREG_test[,-6]   # objeto provisto por ti
kaggle_x_mat <- predict(dv, newdata = kaggle_raw %>% dplyr::select(-ID)) %>% as.matrix()
# si la tabla de test no tiene Exited, entonces ajusta la selección anterior. Si falla, simplemente:
# kaggle_x_mat <- predict(dv, newdata = kaggle_raw %>% dplyr::select(-ID)) %>% as.matrix()

# Vector de labels binarios para glmnet/lightgbm
y_train_num <- ifelse(dataTrain$Exited == "Yes", 1L, 0L)
y_test_num  <- ifelse(dataTest$Exited  == "Yes", 1L, 0L)

# Calculo de pesos/ratio para usar en LASSO y LightGBM
n_pos <- sum(y_train_num == 1)
n_neg <- sum(y_train_num == 0)
scale_pos_weight <- ifelse(n_pos == 0, 1, n_neg / n_pos)
class_weights <- c("0" = 1, "1" = n_neg / n_pos)  # para glmnet usaremos vector de weights

###############################################################################
# 2) MODELO: Logistic LASSO (cv.glmnet) - OJO: usaremos cv.glmnet con weights
###############################################################################
library(glmnet)
# cv.glmnet espera una matriz esparsa a veces - convertir a Matrix si grande
x_train_sparse <- Matrix::Matrix(x_train_mat, sparse = TRUE)

# Hacemos cv.glmnet con family binomial y weights (balanceo)
cv_lasso <- cv.glmnet(
  x = x_train_sparse,
  y = y_train_num,
  family = "binomial",
  alpha = 1,           # LASSO
  nfolds = 5,
  type.measure = "class",
  weights = ifelse(y_train_num == 1, class_weights["1"], class_weights["0"])
)
best_lambda <- cv_lasso$lambda.min
best_lambda

# Probabilidades del lasso sobre el conjunto test (más adelante usaremos esto para OOF y stacking)
x_test_sparse <- Matrix::Matrix(x_test_mat, sparse = TRUE)
lasso_prob_test <- predict(cv_lasso, newx = x_test_sparse, s = "lambda.min", type = "response")[,1]

###############################################################################
# 3) MODELO: Naive Bayes (usando tu pipeline con caret y sampling = "smote")
#    Entrenamiento final sobre dataTrain completo (factores permitidos)
###############################################################################
# Ajusto tune grid como en tu código original (puedes modificar)
grid_nb <- expand.grid(
  usekernel = c(TRUE, FALSE),
  laplace = c(0, 1, 2),
  adjust = c(0, 1, 2)
)

# Entreno NB (este será el modelo final NB)
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
# LightGBM necesita matrix numérica; usamos x_train_mat
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
#    Generamos OOF probs para cada base model (LASSO, NB, LGBM)
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
  
  # --- NB (entrenar naive_bayes::naive_bayes directamente, sin SMOTE para OOF)
  nb_fit <- naivebayes::naive_bayes(Exited ~ ., data = dtr)
  nb_preds <- predict(nb_fit, newdata = dval, type = "prob")[, "Yes"]
  oof_nb[val_idx] <- nb_preds
  
  # --- LASSO (usar model.matrix/dv)
  x_dtr <- predict(dv, newdata = dtr %>% dplyr::select(-Exited)) %>% as.matrix()
  x_dval <- predict(dv, newdata = dval %>% dplyr::select(-Exited)) %>% as.matrix()
  y_dtr <- ifelse(dtr$Exited == "Yes", 1L, 0L)
  
  # cv.glmnet en fold training
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
  
  # --- LightGBM (entrenado en fold train)
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
# Predicciones del meta-modelo sobre el conjunto test
# Para eso necesitamos probabilidades de los modelos finales entrenados sobre todo train:
#  - modelo_nb_final  (caret)
#  - cv_lasso         (cv.glmnet)
#  - lgb_model_final  (lgb)
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

# Optimizo thresholds para cada base y para el meta
thr_nb   <- find_best_threshold(nb_prob_test_full, dataTest$Exited)
thr_lasso <- find_best_threshold(lasso_prob_test_full, dataTest$Exited)
thr_lgb  <- find_best_threshold(lgb_prob_test_full, dataTest$Exited)
thr_meta <- find_best_threshold(meta_prob_test, dataTest$Exited)

thr_nb$best_threshold; thr_nb$best_f1
thr_lasso$best_threshold; thr_lasso$best_f1
thr_lgb$best_threshold; thr_lgb$best_f1
thr_meta$best_threshold; thr_meta$best_f1

###############################################################################
# 7) RESULTADOS: métricas y confusion matrices para cada modelo (usando su mejor umbral)
###############################################################################
# Función de ayuda para CM y métricas
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

# Imprimir resumen
cat("=== RESULTADOS (usando umbral óptimo en test) ===\n")
cat("NB:   best_thr=", thr_nb$best_threshold, " F1=", res_nb$F1, "\n")
print(res_nb$confusion)
cat("LASSO: best_thr=", thr_lasso$best_threshold, " F1=", res_lasso$F1, "\n")
print(res_lasso$confusion)
cat("LGBM: best_thr=", thr_lgb$best_threshold, " F1=", res_lgb$F1, "\n")
print(res_lgb$confusion)
cat("META: best_thr=", thr_meta$best_threshold, " F1=", res_meta$F1, "\n")
print(res_meta$confusion)

###############################################################################
# 8) SUBMISSION PARA KAGGLE (usamos el meta-modelo + umbral óptimo del meta)
###############################################################################
# Preparamos inputs para Kaggle: probabilidades de cada modelo sobre kaggle_raw
# NB necesita los mismos features en formato factor as original -> aplicamos predict sobre kaggle_raw
nb_prob_kaggle <- predict(modelo_nb_final, newdata = kaggle_raw, type = "prob")[, "Yes"]

# Para LASSO y LGBM usamos las matrices dummy (kaggle_x_mat)
# Asegúrate de que kaggle_x_mat tiene las mismas columnas; si no coincide, hay que alinear columnas.
# Si hay columnas que faltan en el kaggle set, rellena con 0:
common_cols <- colnames(x_train_mat)
if(!all(common_cols %in% colnames(kaggle_x_mat))) {
  # Añadir columnas que faltan (cero)
  missing_cols <- setdiff(common_cols, colnames(kaggle_x_mat))
  if(length(missing_cols)) {
    add_mat <- matrix(0, nrow = nrow(kaggle_x_mat), ncol = length(missing_cols),
                      dimnames = list(NULL, missing_cols))
    kaggle_x_mat <- cbind(kaggle_x_mat, add_mat)
  }
}
# Reorder columns to match training
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

# Aplicar el umbral óptimo del meta encontrado en el test
best_meta_thr <- thr_meta$best_threshold
pred_kaggle <- ifelse(meta_prob_kaggle > best_meta_thr, "Yes", "No")

submission <- data.frame(
  ID = kaggle_raw$ID,
  Exited = pred_kaggle
)

# Guardar CSV
write.csv(submission, file = "submission_meta_model.csv", row.names = FALSE)
cat("Submission saved to submission_meta_model.csv (threshold meta:", best_meta_thr, ")\n")
