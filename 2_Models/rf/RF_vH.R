set.seed(123)
library(smotefamily)
library(MLmetrics)
library(caret)
library(randomForest)

data = dataAREG_final[-c(1,2)]
data$Age2 = data$Age^2
data$Age3 = data$Age^3
data$Age3 = data$Age^4

dataAREG_test_final$Age2 = dataAREG_test_final$Age^2
dataAREG_test_final$Age3 = dataAREG_test_final$Age^3
dataAREG_test_final$Age4 = dataAREG_test_final$Age^4



Index <- sample(1:nrow(data), size = nrow(data)*0.8)
dataTrain <- data[Index, ]
dataTest  <- data[-Index, ]

# F1
f1 <- function(data, lev = NULL, model = NULL) {
  f1_val <- MLmetrics::F1_Score(y_pred = data$pred, y_true = data$obs, positive = "Yes")
  c(F1 = f1_val)
}

dataTrain$Exited <- factor(dataTrain$Exited, levels = c(0,1), labels = c("No", "Yes"))
dataTest$Exited  <- factor(dataTest$Exited,  levels = c(0,1), labels = c("No", "Yes"))

control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = f1,
  sampling = "smote",
  verboseIter = TRUE
)


malla<- expand.grid(
  .mtry = c(2, 3, 4,10,13)
)

# Entrenar el model Random Forest
# method = "rf" indica que utilitzi la implementació de randomForest (per defecte)
model_rf <- train(
  Exited ~ .,     
  data = dataTrain,      
  method = "rf",           
  metric = "F1",      
  tuneGrid = malla,  
  trControl = control, 
  importance = TRUE        
)

plot(model_rf)
plot(varImp(model_rf))

prob = predict(model_rf, newdata = dataTrain, type ="prob")

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

best_thr = find_best_threshold(prob[2],dataTrain$Exited)$best_threshold

probTest = predict(model_rf,newdata = dataTest, type="prob")[2]
predTest = as.factor(ifelse(probTest > 0.3, "Yes", "No"))

MLmetrics::F1_Score(predTest, dataTest$Exited)

cm =confusionMatrix(predTest,dataTest$Exited,positive="Yes")
cm

# =============================== PARA KAGGLE ==================================
prob_test <- predict(model_rf, newdata = dataAREG_test_final, type = "prob")[2]
pred_test = as.factor(ifelse(prob_test > 0.3, "Yes", "No"))

submission <- data.frame(
  ID = dataAREG_test_final$ID,
  Exited = pred_test
)

write.csv(submission, "submission_RF_smote_selec_2.csv", row.names=FALSE)
