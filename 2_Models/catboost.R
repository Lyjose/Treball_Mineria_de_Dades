library(catboost)
library(dplyr)
library(MLmetrics)

set.seed(123)

###############################
### 1. PREPARAR LOS DATOS  ####
###############################

data <- data_imputed_AREG[,-c(7,18)]
Index <- sample(1:nrow(data), size = nrow(data)*0.8)
dataTrain <- data[Index, ]
dataTest  <- data[-Index, ]

# Convertir caracteres a factores
dataTrain <- dataTrain %>% mutate(across(where(is.character), as.factor))
dataTest  <- dataTest %>% mutate(across(where(is.character), as.factor))

# Alinear niveles entre train y test
for(col in names(dataTrain)) {
  if(is.factor(dataTrain[[col]])) {
    dataTest[[col]] <- factor(dataTest[[col]], levels = levels(dataTrain[[col]]))
  }
}

# Variable objetivo numérica
dataTrain$Exited_num <- ifelse(dataTrain$Exited == "Yes" | dataTrain$Exited == "1", 1, 0)
dataTest$Exited_num  <- ifelse(dataTest$Exited == "Yes" | dataTest$Exited == "1", 1, 0)

# Crear pool completo
train_pool <- catboost.load_pool(
  data = dataTrain %>% select(-Exited, -Exited_num),
  label = dataTrain$Exited_num
)

test_pool <- catboost.load_pool(
  data = dataTest %>% select(-Exited, -Exited_num),
  label = dataTest$Exited_num
)

###############################
### 2. GRID DE HIPERPARAMS ####
###############################

grid <- expand.grid(
  depth = c(4,6,8),
  learning_rate = c(0.03,0.05,0.1),
  l2_leaf_reg = c(1,3,5)
)

###############################
### 3. CROSS-VALIDATION RÁPIDA
###############################

library(purrr)

results <- purrr::pmap_dfr(
  list(
    depth = as.numeric(grid$depth),
    learning_rate = as.numeric(grid$learning_rate),
    l2_leaf_reg = as.numeric(grid$l2_leaf_reg)
  ),
  function(depth, learning_rate, l2_leaf_reg){
    
    params <- list(
      loss_function = "Logloss",
      eval_metric = "F1",
      iterations = 250,
      depth = depth,
      learning_rate = learning_rate,
      l2_leaf_reg = l2_leaf_reg,
      random_seed = 42,
      border_count = 254
    )
    
    cv <- catboost.cv(
      pool = train_pool,
      params = params,
      fold_count = 5,
      type = "Classical",
      stratified = TRUE,
      partition_random_seed = 42,
      early_stopping_rounds = 30
    )
    
    data.frame(
      depth = depth,
      learning_rate = learning_rate,
      l2_leaf_reg = l2_leaf_reg,
      F1_mean = max(cv$test.F1.mean)
    )
  }
)


# Mejor hiperparámetro
best_params <- results[which.max(results$F1_mean), ]
print(best_params)

###############################
### 4. MODELO FINAL
###############################

final_params <- list(
  loss_function = "Logloss",
  eval_metric = "F1",
  iterations = 1500,  # más iteraciones para modelo final
  depth = best_params$depth,
  learning_rate = best_params$learning_rate,
  l2_leaf_reg = best_params$l2_leaf_reg,
  random_seed = 123,
  border_count = 254,
  early_stopping_rounds = 50
)

modelo_catboost <- catboost.train(
  learn_pool = train_pool,
  test_pool = NULL,
  params = final_params
)

###############################
### 5. MEJOR THRESHOLD F1
###############################

pred_prob <- catboost.predict(
  modelo_catboost,
  test_pool,
  prediction_type = "Probability"
)

thresholds <- seq(0.01, 0.99, by = 0.01)

f1_scores <- sapply(thresholds, function(th){
  preds <- ifelse(pred_prob > th, 1, 0)
  MLmetrics::F1_Score(y_pred = preds, y_true = dataTest$Exited_num, positive = 1)
})

best_threshold <- thresholds[which.max(f1_scores)]
cat("Mejor threshold F1:", best_threshold, "\n")

best_threshold2 <- thresholds[which.max(recall)]
cat("Mejor threshold Recall:", best_threshold2, "\n")

# Predicciones finales con threshold óptimo
preds_test <- ifelse(pred_prob > best_threshold, 1, 0)

cm_best <- caret::confusionMatrix(
  factor(preds_test, levels = c(0,1)),
  factor(dataTest$Exited_num, levels = c(0,1)),
  positive = "0"
)
print(cm_best)

MLmetrics::F1_Score(preds_test,dataTest$Exited_num)


################################################################################
#####                            Para Kaggle                                ####
################################################################################

# 1. Obtener probabilidades de test
Test = data_imputed_AREG_test[-6]
test_pool_kaggle <- catboost.load_pool(
  data = Test
)

probs_test <- catboost.predict(
  modelo_catboost,
  test_pool_kaggle,
  prediction_type = "Probability"
)
# 2. Aplicar el mejor threshold encontrado
pred_test <- ifelse(probs_test > best_threshold, "Yes", "No")


# 3. Crear el dataframe de submission
submission <- data.frame(
  ID = data_imputed_AREG_test$ID,
  Exited = pred_test
)

# 4. Guardar el CSV
write.csv(submission, "submission_catboost.csv", row.names = FALSE)

