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
  nrounds = c(200, 300, 400),          # Més iteracions sol anar millor si eta és baix
  max_depth = c(2,4, 6),         # Provem arbres més profunds
  eta = c(0.04, 0.01, 0.05),             # Velocitat d'aprenentatge més fina
  gamma = 0,
  colsample_bytree = c(0.8),
  min_child_weight = c(1, 3),     # Ajuda a regularitzar
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
plot(importancia, top = 20, main = "Ranking de Variables més Importants (XGBoost)")

# 4. EXTRACCIÓ DE LES PITJORS (SOROLL)
# Això t'ajuda a saber què eliminar per millorar l'F1
imp_df <- importancia$importance
imp_df$Variable <- rownames(imp_df)
imp_df <- imp_df[order(imp_df$Overall, decreasing = FALSE), ] # Ordenem de pitjor a millor

cat("\n--- LES 10 VARIABLES MENYS IMPORTANTS (Possibles Candidates a Esborrar) ---\n")
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

