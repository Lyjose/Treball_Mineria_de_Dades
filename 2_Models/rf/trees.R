packages <- c("DAAG", "caret", "party", "rpart", "rpart.plot","mlbench",
              "pROC","tree","C50","printr","ggplot2")

install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

lapply(packages, install_if_missing)

load("data_NA_imputed_AREG_test.RData")
load("dataAREG_outliers.RData")

data_test <- data_imputed_AREG_test[-6]
mydata <- dataAREG

clases <- sapply(data, class)

set.seed(123)
ind <- sample(1:nrow(mydata), 0.8*nrow(mydata))
train <- mydata[ind,]
test <- mydata[-ind,]

train$Exited <- factor(train$Exited, levels = c(0,1), labels = c("No", "Yes"))
test$Exited  <- factor(test$Exited,  levels = c(0,1), labels = c("No", "Yes"))

# ====================================================================
###                          MÉTODO DANTE                         ####
# ====================================================================

# CONSTRUIR ARBOL CON CP=0 Y LUEGO ELEGIR CP

tree <- rpart(Exited ~ ., data = train, cp = 0)
printcp(tree)
plotcp(tree)
xerror <- tree$cptable[,"xerror"]

imin.xerror <- which.min(xerror) # cp con menor error 0.00615114
imin.xerror
tree$cptable[imin.xerror, ]
upper.xerror <- xerror[imin.xerror] + tree$cptable[imin.xerror, "xstd"] 
upper.xerror
tree <- prune(tree, cp = 0.00615) 
rpart.plot(tree)

importance <- tree$variable.importance # Equivalente a caret::varImp(tree) 
importance <- round(100*importance/sum(importance), 1)
importance

# para graficar vars

df_imp <- data.frame(
  Variable = names(importance),
  Valor    = importance
)

ggplot(df_imp, aes(x = reorder(Variable, Valor), y = Valor)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  geom_text(aes(label = Valor), hjust = -0.2) +
  labs(
    title = "Importància de Variables (Decision trees)",
    x = "Variables",
    y = "Importancia Relativa (%)"
  ) +
  theme_minimal()

# Predicciones

probs <- predict(tree, test, type = "prob")
predictions<- as.factor(predictions<- ifelse(probs>0.3,"Yes","No")[,"Yes"])
cm <- confusionMatrix(predictions, test$Exited, positive="Yes")

tp <- cm$table[2,2]
fp <- cm$table[2,1]
fn <- cm$table[1,2]

precision <- tp/(tp+fp)
recall_1 <- tp/(tp+fn)
f1_1 <- 2*precision*recall_1/(precision+recall_1)
Accuracy_1 = cm$overall["Accuracy"]
Specificity = cm$byClass["Specificity"]


# prediccions per test (submission)
pred_test <- predict(tree, newdata = data_test, type = "class")

submission_tree1 <- data.frame(
  ID = data_test$ID,
  Exited = pred_test
)

write.csv(submission_tree1, "submission_tree1.csv", row.names = FALSE)



# ====================================================================
###                          BUSCAR ARBOL OPTIMO                 ####
# ====================================================================

tree2 <- tree(
  formula = Exited ~ .,
  data    = train,
  mindev  = 0
)

set.seed(123)
cv_arbol <- cv.tree(tree2, FUN = prune.misclass, K = 5)
size_optimo <- rev(cv_arbol$size)[which.min(rev(cv_arbol$dev))]
size_optimo
resultados_cv <- data.frame(n_nodos = cv_arbol$size, clas_error = cv_arbol$dev,
                            alpha = cv_arbol$k)
resultados_cv
arbol_final <- prune.misclass(
  tree = tree2,
  best = size_optimo)


p2 <- predict(arbol_final, test, type = "vector")
predictions<- as.factor(predictions<- ifelse(p2>0.2,"Yes","No")[,"Yes"])
cm2 <- confusionMatrix(predictions, test$Exited, positive="Yes")

tp <- cm2$table[2,2]
fp <- cm2$table[2,1]
fn <- cm2$table[1,2]

precision <- tp/(tp+fp)
recall_2 <- tp/(tp+fn)
f1_2 <- 2*precision*recall_2/(precision+recall_2)
Accuracy_2 = cm2$overall["Accuracy"]
Specificity_2 = cm2$byClass["Specificity"]

pred_test <- predict(tree2, newdata = data_test, type = "class")

submission_tree2 <- data.frame(
  ID = data_test$ID,
  Exited = pred_test
)

write.csv(submission_tree1, "submission_tree2.csv", row.names = FALSE)


# ====================================================================
###                            ARBOL CON CARET                    ####
# ====================================================================


p1 <- predict(tree, test, type = 'prob')
head(p1)
p1 <- p1[,2]
r <- multiclass.roc(test$Exited, p1, percent = TRUE)
roc <- r[['rocs']]
r1 <- roc[[1]]
plot.roc(r1,print.auc=TRUE,
         auc.polygon=TRUE,
         grid=c(0.1, 0.2),
         grid.col=c("green", "red"),
         max.auc.polygon=TRUE,
         auc.polygon.col="lightblue",
         print.thres=TRUE,
         main= 'ROC Curve')

# a partir de 0.2 clasificará como 1.

obs <- test$Exited
caret::postResample(p2, obs)

### Modelling with package caret 3rd strategy by using Caret
caret.rpart <- train(Exited ~ ., method = "rpart", data = train, 
                     tuneLength = 20,
                     trControl = trainControl(method = "cv", number = 10)) 
ggplot(caret.rpart)
rpart.plot(caret.rpart$finalModel)

caret.rpart <- train(Exited ~ ., method = "rpart", data = train, 
                     tuneLength = 20,
                     trControl = trainControl(method = "cv", number = 10,
                                              selectionFunction = "oneSE")) 
rpart.plot(caret.rpart$finalModel)
var.imp <- varImp(caret.rpart)
plot(var.imp)

prob2 <- predict(caret.rpart, test, type = "prob")
predictions<- as.factor(predictions<- ifelse(prob2>0.2,"Yes","No")[,"Yes"])

cm3 <- confusionMatrix(predictions, test$Exited, positive="Yes")

tp <- cm3$table[2,2]
fp <- cm3$table[2,1]
fn <- cm3$table[1,2]

precision <- tp/(tp+fp)
recall_3 <- tp/(tp+fn)
f1_3 <- 2*precision*recall_3/(precision+recall_3)
Accuracy_3 = cm3$overall["Accuracy"]
Specificity_3 = cm3$byClass["Specificity"]


# ====================================================================
###                     MÉTODO DANTE CON SMOTE                   ####
# ====================================================================

f1 <- function(data, lev = NULL, model = NULL) {
  f1_val <- MLmetrics::F1_Score(y_pred = data$pred, y_true = data$obs, positive = "Yes")
  c(F1 = f1_val)
}

ctrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 10,
  classProbs = TRUE,
  summaryFunction = f1,
  sampling = "smote",
  verboseIter = TRUE
)

set.seed(123)
tree_caret <- train(Exited ~ .,data = train, method = "rpart", trControl = ctrl,          # Aquí sí va tu objeto ctrl
  metric = "F1", tuneLength = 20)


rpart.plot(tree_caret$finalModel)
var.imp <- varImp(tree_caret)
plot(var.imp)

# Mirem millor threshold

prob4 <- predict(tree_caret, test, type = "prob")[, "Yes"]

thresholds <- seq(0.01, 0.8, by = 0.01)
f1_scores <- numeric(length(thresholds))

for (i in seq_along(thresholds)) {
  threshold <- thresholds[i]
  preds <- ifelse(prob4 > threshold, "Yes", "No")
  f1_scores[i] <- MLmetrics::F1_Score(y_pred = preds, y_true = test$Exited, positive = "Yes")
}

threshold_f1_df <- data.frame(Threshold = thresholds, F1_Score = f1_scores)

print(threshold_f1_df)

best_threshold <- threshold_f1_df$Threshold[which.max(threshold_f1_df$F1_Score)]
best_threshold

predictions <- ifelse(prob4 > best_threshold, "Yes", "No")
# CM
cm_best <- confusionMatrix(
  factor(predictions, levels = c("No", "Yes")),
  test$Exited,
  positive = "Yes"
); print(cm_best)

MLmetrics::F1_Score(test$Exited,predictions,"Yes")

