library(MLmetrics)
library(dplyr)
library(glmnet)
library(caret)
library(doParallel)
library(splines)

data = dataAREG_final[,-c(1,2)]

############################ Descriptiva #####################################

vars = colnames(data)[-21]
par(mfrow = c(4,5), mar = c(3,3,3,1))
for (va in vars){
  if (!is.factor(data[,va])){
    boxplot(as.formula(paste0(va,"~Exited")),data,main=va,col=c(2,3),horizontal=T)
  } else{
    barplot(prop.table(table(data$Exited, data[,va]),2),main=va,col=c(2,3))
  }
}

############################ Modelatge #####################################

set.seed(123)

Index <- sample(1:nrow(data), size = nrow(data)*0.8)
dataTrain <- data[Index, ]
dataTest  <- data[-Index, ]

dataTrain$Exited <- factor(dataTrain$Exited, levels = c(0,1), labels = c("No", "Yes"))
dataTest$Exited  <- factor(dataTest$Exited,  levels = c(0,1), labels = c("No", "Yes"))

# F1
f1 <- function(data, lev = NULL, model = NULL) {
  f1_val <- MLmetrics::F1_Score(y_pred = data$pred, y_true = data$obs, positive = "Yes")
  c(F1 = f1_val)
}

############################ Model inicial #####################################

mod1 = glm(Exited ~ ., family = binomial(link="logit"),dataTrain)
mod2 = glm(Exited ~ ., family = binomial(link="probit"),dataTrain)
mod3 = glm(Exited ~ ., family = binomial(link="cloglog"),dataTrain)

deviance(mod1);deviance(mod2);deviance(mod3)
AIC(mod1);AIC(mod2);AIC(mod3)

# ES MILLOR LINK LOGIT
par(mfrow=c(2,2))
plot(mod1)
plot(mod2)
plot(mod3)

prob1 = predict(mod1, newdata = dataTest, type="response")
pred1 = ifelse( prob1 > 0.2,"Yes","No")

MLmetrics::F1_Score(dataTest$Exited,pred1)
MLmetrics::Sensitivity(dataTest$Exited,pred1)

############################ Modelatge amb splines #############################

mod_splines <- glm(
  Exited ~  
    # Variables continuas con splines (3-5 grados de libertad)
    ns(Age, df = 4) +                    # Relación en U con edad
    ns(CreditScore, df = 3) +           
    ns(Balance, df = 2) +                # Balance puede tener efectos no lineales
    ns(Tenure, df = 3) +                 # Tenure al cuadrado típicamente
    ns(NetPromoterScore, df = 3) +
    ns(TransactionFrequency, df = 3) +
    ns(AvgTransactionAmount, df = 3) +
    ns(DigitalEngagementScore, df = 3) +
    ns(EstimatedSalary,df=3) +                    # Linear si no hay evidencia
    
    # Variables categóricas
    Gender + EducationLevel + LoanStatus + Geography + 
    ComplaintsCount + HasCrCard + IsActiveMember + 
    CustomerSegment + MaritalStatus + SavingsAccountFlag + 
    NumOfProducts,
  
  family = binomial(link = "logit"),
  data = dataTrain
)


prob2 = predict(mod_splines, newdata = dataTest, type="response")
pred2 = ifelse( prob2 > 0.5,"Yes","No")

MLmetrics::F1_Score(dataTest$Exited,pred2)
MLmetrics::Sensitivity(dataTest$Exited,pred2)

plot(mod_splines)

summary(mod_splines)
AIC(mod1);AIC(mod_splines)

############################ Modelatge amb splines #############################


mod_splines2 <- glm(
  Exited ~
    # Variables continuas con splines (3-5 grados de libertad)
    ns(Age, df = 4) +                    # Relación en U con edad
    ns(NetPromoterScore, df = 1) +

    Gender + EducationLevel + Geography + 
    IsActiveMember + NumOfProducts,
  
  family = binomial(link = "logit"),
  data = dataTrain
)

AIC(mod1);AIC(mod_splines);AIC(mod_splines2)

plot(mod_splines2)

summary(mod_splines2)

prob3 = predict(mod_splines2, newdata = dataTest, type="response")
pred3= ifelse( prob3 > 0.1,"Yes","No")

MLmetrics::F1_Score(dataTest$Exited,pred3)
MLmetrics::Specificity(dataTest$Exited,pred3)


ConfusionMatrix(pred3,dataTest$Exited)

######################### ELASTIC NET #########################################
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Fórmula con splines
formula_completa <- Exited ~ 
  ns(Age, df = 4) +
  ns(CreditScore, df = 2) +
  ns(Balance, df = 2) +
  ns(Tenure, df = 1) +
  ns(NetPromoterScore, df = 3) +
  ns(TransactionFrequency, df = 1) +
  ns(AvgTransactionAmount, df = 1) +
  ns(DigitalEngagementScore, df = 1) +
  EstimatedSalary +
  Gender + EducationLevel + LoanStatus + Geography + 
  ComplaintsCount + HasCrCard + IsActiveMember + 
  CustomerSegment + MaritalStatus + SavingsAccountFlag + 
  NumOfProducts

# TODAS las interacciones de orden 2
formula_interacciones <- update(formula_completa, ~ (.)^2)


# Configuración CV optimizando F1
train_control_f1 <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = f1,     
  verboseIter = TRUE,
  allowParallel = TRUE
)

# Grid de hiperparámetros
elastic_grid <- expand.grid(
  alpha = seq(0, 1, by = 0.2),
  lambda = exp(seq(-4, 0, length.out = 10))
)

elastic_f1 <- train(
  formula_interacciones,
  data = dataTrain,
  method = "glmnet",
  trControl = train_control_f1,
  tuneGrid = elastic_grid,
  metric = "F1",                    # OPTIMIZAR F1
  family = "binomial"
)


prob3 = predict(elastic_f1, newdata = dataTest, type="prob")
pred3 = ifelse( prob3 > 0.25,"Yes","No")[,2]

MLmetrics::F1_Score(dataTest$Exited,pred3)
MLmetrics::Specificity(dataTest$Exited,pred3)

################################################################################
#####                            THRESHOLD                                ####
################################################################################

thresholds <- seq(0.01, 0.8, by = 0.01)

# Inicializar un vector para almacenar los resultados del F1
f1_scores <- numeric(length(thresholds))

# Loop para probar cada threshold y calcular el F1
for (i in seq_along(thresholds)) {
  threshold <- thresholds[i]
  preds <- ifelse(prob3 > threshold, "Yes", "No")
  f1_scores[i] <- MLmetrics::F1_Score(y_pred = preds[,2], y_true = dataTest$Exited, positive = "Yes")
}

threshold_f1_df <- data.frame(Threshold = thresholds, F1_Score = f1_scores)

print(threshold_f1_df)

best_threshold <- threshold_f1_df$Threshold[which.max(threshold_f1_df$F1_Score)]
best_threshold

preds_testtrain <- ifelse(prob3 > best_threshold, "Yes", "No")[,2]
# CM
cm_best <- confusionMatrix(
  factor(preds_testtrain, levels = c("No", "Yes")),
  dataTest$Exited,
  positive = "Yes"
); print(cm_best)

MLmetrics::F1_Score(dataTest$Exited,preds_testtrain,"Yes")


################################################################################
#####                            Para Kaggle                                ####
################################################################################

# 1. Obtener probabilidades de test
probs_test <- predict(elastic_f1, newdata = data_imputed_AREG_test[-6], type = "prob")[, "Yes"]

pred_test <- ifelse(probs_test > 0.25, "Yes", "No")


# 3. Crear el dataframe de submission
submission <- data.frame(
  ID = data_imputed_AREG_test$ID,
  Exited = pred_test
)

# 4. Guardar el CSV
write.csv(submission, "submission_glm.csv", row.names = FALSE)

