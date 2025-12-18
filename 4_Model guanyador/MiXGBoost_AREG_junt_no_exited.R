# ==============================================================================
# XGBOOST OPTIMITZAT (Threshold Tuning)
# ==============================================================================

# 1. LLIBRERIES I SEED
packages <- c("caret", "xgboost", "smotefamily", "MLmetrics", "dplyr", "mpae", "gbm")
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) install.packages(pkg)
  library(pkg, character.only = TRUE)
}

set.seed(123)

# 2. CARREGAR I PREPARAR DADES
# -----------------------------------------------------------
# Carreguem el fitxer FINAL netejat d'outliers i brossa
load("data_Final_AREG_train.RData") # O el nom que tinguis (data_Final_FOREST_train.RData, etc.)

# Assignem a mydata (La variable que es diu dataAREG_final dins l'RData)
if(exists("dataAREG_final")) { 
  mydata <- dataAREG_final 
} else { 
  print('error')
}

# 3. PREPARACIÓ
# ------------------------------------------------------------------------------
# IMPORTANT: Eliminem ID i Surname si encara hi són (no volem entrenar amb noms)
if("ID" %in% names(mydata)) mydata$ID <- NULL
if("Surname" %in% names(mydata)) mydata$Surname <- NULL

set.seed(123)

# createDataPartition manté la proporció de 'Exited'
# p = 0.8 significa 80% per al train
Index <- createDataPartition(mydata$Exited, p = 0.8, list = FALSE)

dataTrain <- mydata[Index, ]
dataTest  <- mydata[-Index, ]

# Etiquetes i Funció F1
dataTrain$Exited <- factor(dataTrain$Exited, levels = c(0,1), labels = c("No", "Yes"))
dataTest$Exited  <- factor(dataTest$Exited,  levels = c(0,1), labels = c("No", "Yes"))

f1 <- function(data, lev = NULL, model = NULL) {
  f1_val <- MLmetrics::F1_Score(y_pred = data$pred, y_true = data$obs, positive = "Yes")
  c(F1 = f1_val)
}

# 4. CONTROL I GRID (Aquí millorem paràmetres)
# ------------------------------------------------------------------------------
control <- trainControl(
  method = "repeatedcv", # Cross Validation amb repeats>1
  number = 5, # Divideix el data_train en 5 troços -> 4 train (80%) i 1 validation (20%)
  repeats = 2, # Fa el procés 2 vegades (amb altres divisions)
  classProbs = TRUE,
  summaryFunction = f1,
  sampling = "smote", 
  verboseIter = TRUE
)

# GRID AMPLIAT: Més opcions per trobar el millor model
xgb_grid <- expand.grid(
  nrounds = c(500),          # c(500, 1000)
  max_depth = c(3),         # c(3, 5, 7)
  eta = c(0.01),             # c(0.01, 0.02)
  gamma = 0,
  colsample_bytree = c(0.8),# c(0.6, 0.8)
  min_child_weight = c(1),     # c(1, 3, 5)
  subsample = 0.8
)

# 5. ENTRENAMENT
# ------------------------------------------------------------------------------
cat("Entrenant XGBoost millorat...\n")
model_xgb <- train(
  Exited ~ ., 
  data = dataTrain,
  method = "xgbTree",
  trControl = control,
  metric = "F1",
  tuneGrid = xgb_grid
)

print(model_xgb)

# ==============================================================================
# 8. ANÀLISI D'IMPORTÀNCIA DE VARIABLES (XGBoost)
# ==============================================================================

# 1. Calculem la importància
importancia <- varImp(model_xgb, scale = FALSE)

# 2. Mostrem la taula per consola
print(importancia)

# 3. Gràfic Professional (Per al Report)
# Aquest gràfic mostra quines variables aporten més a la predicció
plot(importancia, main = "Ranking de Variables més Importants (XGBoost)")

# 4. EXTRACCIÓ DE LES PITJORS (SOROLL)
# Això t'ajuda a saber què eliminar per millorar l'F1
imp_df <- importancia$importance
imp_df$Variable <- rownames(imp_df)
imp_df <- imp_df[order(imp_df$Overall, decreasing = FALSE), ] # Ordenem de pitjor a millor

cat("\n--- LES 5 VARIABLES MENYS IMPORTANTS (Possibles Candidates a Esborrar) ---\n")
print(head(imp_df, 5))

# 6. VALIDACIÓ I OPTIMITZACIÓ DE LLINDAR (LA CLAU)
# ------------------------------------------------------------------------------
cat("\n--- BUSCANT EL MILLOR TALL (Cerca Fina) ---\n")

probs_val <- predict(model_xgb, newdata = dataTest, type = "prob")[,"Yes"]

# ESTRATÈGIA: Busquem només al voltant del 0.27 (+- 0.15)
# Això estalvia càlculs inútils
talls <- seq(0.15, 0.5, by = 0.005) # Passem a 0.005 per tenir més precisió!

millor_f1 <- 0
millor_tall <- 0.27 # Valor per defecte segur

for(t in talls) {
  pred_temp <- factor(ifelse(probs_val > t, "Yes", "No"), levels=c("No","Yes"))
  
  # Try-catch per evitar errors si un tall no prediu cap "Yes"
  try({
    f1_val <- MLmetrics::F1_Score(y_pred = pred_temp, y_true = dataTest$Exited, positive = "Yes")
    
    if(!is.na(f1_val) && f1_val > millor_f1) {
      millor_f1 <- f1_val
      millor_tall <- t
    }
  }, silent=TRUE)
}

cat("------------------------------------------------\n")
cat("NOU F1 ÒPTIM:", millor_f1, "\n")
cat("MILLOR TALL:", millor_tall, "\n")
cat("------------------------------------------------\n")


pred <- factor(ifelse(probs_val > 0.5, "Yes", "No"), levels=c("No","Yes"))

confusionMatrix(dataTest$Exited,pred,positive="Yes")
MLmetrics::F1_Score(dataTest$Exited,pred,positive="Yes")

########### GRÀFIC
prob <- predict(model_xgb, dataTest, type = "prob")[, "Yes"]

# thresholds
thresholds <- seq(0, 1, by = 0.01)

recall <- numeric(length(thresholds))
f1     <- numeric(length(thresholds))

for (i in seq_along(thresholds)) {
  
  t <- thresholds[i]
  
  pred_t <- factor(
    ifelse(prob >= t, "Yes", "No"),
    levels = c("No", "Yes")
  )
  
  recall[i] <- Recall(
    y_true = dataTest$Exited,
    y_pred = pred_t,
    positive = "Yes"
  )
  
  f1[i] <- F1_Score(
    y_true = dataTest$Exited,
    y_pred = pred_t,
    positive = "Yes"
  )
}

df <- data.frame(
  threshold = thresholds,
  Recall = recall,
  F1 = f1
)

ggplot(df, aes(x = threshold)) +
  geom_line(aes(y = Recall, color = "Recall"), linewidth = 1.1) +
  geom_line(aes(y = F1, color = "F1"), linewidth = 1.1) +
  geom_vline(
    xintercept = 0.3125,
    linetype = "dashed",
    linewidth = 1,
    color = "black"
  ) +
  annotate(
    "text",
    x = 0.3125,
    y = max(c(df$Recall, df$F1), na.rm = TRUE),
    label = "threshold = 0.375",
    hjust = -0.1,
    vjust = 1.2,
    size = 3.5
  ) +
  scale_color_manual(values = c("Recall" = "coral", "F1" = "turquoise")) +
  labs(
    title = "Recall y F1 para Exited según el threshold",
    x = "Threshold",
    y = "Valor",
    color = "Métrica"
  ) +
  theme_minimal()



#===========================
# KAGGLE
#===========================

cat("\n--- GENERANT SUBMISSION FINAL ---\n")

# 1. Carreguem les dades del TEST (les que no tenen Exited)
# Assegura't que el fitxer es diu així (és el que vam generar amb l'script d'Outliers)
load("data_Final_AREG_test.RData") 

if(exists("dataAREG_test_final")) {
  data_kaggle <- dataAREG_test_final
} else {
  # Si el nom és diferent, intentem agafar l'objecte carregat
  # (Busquem un objecte que tingui 'test' al nom o agafem l'últim carregat)
  vars_disponibles <- ls()
  nom_test <- vars_disponibles[grep("test", vars_disponibles, ignore.case = TRUE)]
  if(length(nom_test) > 0) {
    data_kaggle <- get(nom_test[1])
  } else {
    stop("Error: No trobo l'objecte del test carregat.")
  }
}

# 2. Guardem els IDs per al fitxer final (Molt important!)
kaggle_ids <- data_kaggle$ID

# 3. Neteja igual que al Train (sense ID ni Surname)
if("ID" %in% names(data_kaggle)) data_kaggle$ID <- NULL
if("Surname" %in% names(data_kaggle)) data_kaggle$Surname <- NULL
# Si hi ha la columna Exited (encara que sigui NA), la treiem per evitar errors
if("Exited" %in% names(data_kaggle)) data_kaggle$Exited <- NULL

# 4. Predicció de Probabilitats
cat("Predint sobre el test de Kaggle...\n")
probs_kaggle <- predict(model_xgb, newdata = data_kaggle, type = "prob")[,"Yes"]

# 5. Aplicar el MILLOR TALL (trobat al pas anterior)
# Si la probabilitat és > millor_tall -> "Yes", sinó "No"
pred_final <- ifelse(probs_kaggle > millor_tall, "Yes", "No")

# 6. Crear Dataframe i Guardar CSV
submission <- data.frame(
  ID = kaggle_ids,
  Exited = pred_final
)

# Posem el F1 al nom del fitxer per tenir control de versions
nom_fitxer <- paste0("submission_XGB_junt_no_exited.csv")

write.csv(submission, nom_fitxer, row.names = FALSE)

cat("FET! Fitxer guardat com:", nom_fitxer, "\n")
cat("Aquest és el fitxer que has de penjar a Kaggle. Sort! 🚀\n")
