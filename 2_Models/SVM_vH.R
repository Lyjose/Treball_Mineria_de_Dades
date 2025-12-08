set.seed(123)
library(MLmetrics)
library(caret)
library(recipes)
library(dplyr)

data = dataAREG_final[-c(1,2)]
data$Age2 = data$Age^2
data$Age3 = data$Age^3
data$Age4 = data$Age^4

dataAREG_test_final$Age2 = dataAREG_test_final$Age^2
dataAREG_test_final$Age3 = dataAREG_test_final$Age^3
dataAREG_test_final$Age4 = dataAREG_test_final$Age^4



Index <- sample(1:nrow(data), size = nrow(data)*0.8)
dataTrain <- data[Index, ]
dataTest  <- data[-Index, ]
  
  ######### PUNTO 2 ###########
    # F1
f1 <- function(data, lev = NULL, model = NULL) {
    f1_val <- MLmetrics::F1_Score(y_pred = data$pred, y_true = data$obs, positive = "Yes")
    c(F1 = f1_val)
}
  
dataTrain$Exited <- factor(dataTrain$Exited, levels = c(0,1), labels = c("No", "Yes"))
dataTest$Exited  <- factor(dataTest$Exited,  levels = c(0,1), labels = c("No", "Yes"))
  
control <- trainControl(
    method = "repeatedcv",
    number = 10,
    repeats = 10,
    classProbs = TRUE,
    summaryFunction = f1,
    verboseIter = TRUE
  )
  
# 1. Carregar paquets necessaris
library(caret)
library(recipes)
library(kernlab) # Per a SVM
library(pROC)    # Per a la funció de resum prSummary (inclou F1)

# 2. Assegurar la validesa dels noms dels nivells de la variable objectiu
# "No" i "Yes" ja són vàlids, però aquest pas garanteix que la mètrica F1 funcioni
# si hi hagués hagut problemes de noms (com en l'error anterior).
levels(dataTrain$Exited) <- make.names(levels(dataTrain$Exited))

# 3. Definir la Recepta (Pipeline de Preprocessament)

recepta <- 
  recipe(Exited ~ ., data = dataTrain) %>%
  
  # A. Codificació de Categòriques (Geography, Gender, HasCrCard, IsActiveMember)
  # step_dummy converteix tots els factors/nominals a variables 0/1 (One-Hot Encoding)
  step_dummy(all_nominal_predictors(), one_hot = FALSE) %>% # one_hot = FALSE: crea k-1 columnes (evita multicolinealitat perfecta)
  
  # B. Normalització de Numèriques (CreditScore, Age, Age2, Age3, etc.)
  # S'aplica a totes les variables que quedin i siguin numèriques (originals + dummies)
  step_center(all_numeric_predictors()) %>%
  step_scale(all_numeric_predictors())

# 4. Definir el Control de l'Entrenament (incloent la mètrica F1)

# La funció 'prSummary' proporciona la mètrica F1.
control_entrenament <- trainControl(
  method = "repeatedcv",
  number = 5,          # 5-fold Cross-Validation
  repeats = 3,         # Repetida 3 vegades
  summaryFunction = f1, # Utilitza la funció que conté F1
  classProbs = TRUE,   # Necessari per calcular probabilitats (i F1/AUC)
  verboseIter = FALSE
)

# 5. Definir la Malla de Tuning (mtry i C)

# El Kernel Radial (svmRadial) té dos hiperparàmetres principals a optimitzar:
# - C (Cost): Penalització per errors de classificació.
# - sigma: Paràmetre de dispersió del kernel (controla la forma del límit de decisió).
malla_radial <- expand.grid(
  .C = c(0.25), # Prova diferents valors de Cost
  .sigma = c(0.01) # Prova diferents valors de sigma
)

# 6. Entrenar el Model SVM Radial

model_svm <- train(
  recepta,                 # Li passem la recepta de preprocessament
  data = dataTrain,        # Les dades originals
  method = "svmRadial",    # Kernel de Funció de Base Radial (RBF)
  metric = "F1",           # Utilitzem l'F1-score per seleccionar el millor model
  tuneGrid = malla_radial, # La malla d'hiperparàmetres
  trControl = control_entrenament
)

# 7. Inspeccionar els resultats
print(model_svm)
# Mostrar els resultats detallats de cada combinació C i sigma
model_svm$results

# El millor model es troba a:
model_svm$bestTune

pred = predict(model_svm, 
                newdata = dataTest, 
                type = "raw") 



 # PARA KAGGLE
# --- 1. Extreure la Recepta Correctament ---

# Si has entrenat amb la sintaxi 'train(recepta, ...)', la recepta finalitzada 
# (que conté l'ajust de les mitjanes, desviacions i dummies) es guarda aquí:
recepta_entrenada <- model_svm$recipe 

# --- 2. Aplicar la Transformació al Test ---

# Ara 'recepta_entrenada' ja no és NULL i conté la informació necessària per 'bake()'.
data_test_processada <- bake(recepta_entrenada, new_data = dataAREG_test_final)

# --- 3. Predicció Final ---

# Si la teva data ja té les columnes correctament processades, la predicció funcionarà.
pred_test <- predict(model_svm, 
                     newdata = dataAREG_test_final, 
                     type = "raw") 

# --- 4. Generar Submission ---
submission <- data.frame(
  ID = dataAREG_test_final$ID, 
  Exited = pred_test
)

write.csv(submission, "submission_arreglada.csv", row.names = FALSE)