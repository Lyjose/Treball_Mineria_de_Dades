###MAIN PACKAGES
library(DAAG)
library(mlbench)
library(caret)
library(pROC)
library(printr)
library(randomForest)
library(ranger)
library(MLmetrics) # Necessari per F1

# 1. CARREGAR I PREPARAR DADES
# -----------------------------------------------------------
# Carreguem el fitxer FINAL netejat d'outliers i brossa
load("data_Final_AREG_train.RData") # O el nom que tinguis (data_Final_FOREST_train.RData, etc.)

# Assignem a mydata (La variable que es diu dataAREG_final dins l'RData)
if(exists("dataAREG_final")) { 
  mydata <- dataAREG_final 
} else { 
  print('error')
}

# IMPORTANT: Eliminem ID i Surname si encara hi són (no volem entrenar amb noms)
if("ID" %in% names(mydata)) mydata$ID <- NULL
if("Surname" %in% names(mydata)) mydata$Surname <- NULL

set.seed(123)

# createDataPartition manté la proporció de 'Exited'
# p = 0.8 significa 80% per al train
Index <- createDataPartition(mydata$Exited, p = 0.8, list = FALSE)

train <- mydata[Index, ]
test  <- mydata[-Index, ]

# --- COMPROVACIÓ (Opcional, per a la teva tranquil·litat) ---
cat("Proporció Train:\n")
print(prop.table(table(train$Exited)))

cat("Proporció Test:\n")
print(prop.table(table(test$Exited)))

train$Exited <- factor(train$Exited, levels = c(0,1), labels = c("No", "Yes"))
test$Exited  <- factor(test$Exited,  levels = c(0,1), labels = c("No", "Yes"))

#### Modelling with RandomForest ####

### A) Bagging trees (mtry = p)
# -----------------------------------------------------------
bagtrees <- randomForest(Exited ~ ., data = train, mtry = ncol(train) - 1)
# plot(bagtrees, main = "Error Bagging")
# legend("right", colnames(bagtrees$err.rate), lty = 1:5, col = 1:6)
pred2 <- predict(bagtrees, newdata = test)
cat("\n--- BAGGING TREES ---\n")
print(caret::confusionMatrix(pred2, test$Exited, positive="Yes"))

### B) Random Forest Standard
# -----------------------------------------------------------
rf <- randomForest(Exited ~ ., data = train)
# importance(rf) 
# varImpPlot(rf)
pred3 <- predict(rf, newdata = test)
cat("\n--- STANDARD RF ---\n")
print(caret::confusionMatrix(pred3, test$Exited, positive="Yes"))

### C) Random Forest with caret (Tuned)
# -----------------------------------------------------------
# Tunejant mtry segons la regla de l'arrel quadrada
mtry.class <- sqrt(ncol(train) - 1)
tuneGrid <- data.frame(mtry = floor(c(mtry.class/2, mtry.class, 2*mtry.class)))

set.seed(1234)
# Afegim control per si volem optimitzar per Sensitivity després
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE)

rf.caret <- train(Exited ~ ., 
                  data = train, 
                  method = "rf", 
                  tuneGrid = tuneGrid,
                  trControl = ctrl)

cat("\n--- CARET RF TUNED ---\n")
print(rf.caret)
plot(rf.caret)

# Predicció estandard (Tall 0.5)
pred4 <- predict(rf.caret, newdata = test)
print(caret::confusionMatrix(pred4, test$Exited, positive="Yes")) 

# Comprovació d'Overfitting
ptrain <- predict(rf.caret, train, type = 'raw')
cat("\nCheck Overfitting (Train vs Test Accuracy):\n")
cat("Train Acc:", confusionMatrix(ptrain, train$Exited)$overall['Accuracy'], "\n")
cat("Test Acc: ", confusionMatrix(pred4, test$Exited)$overall['Accuracy'], "\n")


# ==============================================================================
# PART NOVA: APLICAR THRESHOLD 0.2 I CLASSIFICAR
# ==============================================================================
# "threshold tiene q ser 0.2"
# "mirar codigo trees que está como hacerlo y dejarlo clasificado"

cat("\n============================================\n")
cat(" APLICANT THRESHOLD 0.2 (Per millorar Sensitivity)\n")
cat("============================================\n")

# 1. Obtenim les PROBABILITATS (no la classe raw)
probs_test <- predict(rf.caret, newdata = test, type = "prob")[,"Yes"]

# 2. Apliquem el tall manual a 0.2
# Si la prob > 0.2, diem que SÍ marxa. Això caçarà molts més fugitius.
threshold <- 0.2
pred_threshold <- factor(ifelse(probs_test > threshold, "Yes", "No"), levels = c("No", "Yes"))

# 3. Matriu de Confusió amb el nou tall
cm_threshold <- confusionMatrix(pred_threshold, test$Exited, positive = "Yes")
print(cm_threshold)

# 4. Càlcul de mètriques finals (postResample i F1)
obs <- test$Exited

# postResample dóna Accuracy i Kappa
resample_metrics <- caret::postResample(pred = pred_threshold, obs = obs)
print(resample_metrics)

# F1 Score (Molt important ja que Accuracy enganya amb el threshold baix)
f1_val <- MLmetrics::F1_Score(y_pred = pred_threshold, y_true = obs, positive = "Yes")
cat("\nF1 SCORE FINAL (Tall 0.2):", f1_val, "\n")

# Gràfic de la corba ROC per veure si 0.2 és realment bo
roc_obj <- roc(test$Exited, probs_test)
plot(roc_obj, print.auc = TRUE, main="ROC Curve Random Forest")
points(coords(roc_obj, x="best", best.method="closest.topleft"), col="red", pch=19) # El punt òptim matemàtic