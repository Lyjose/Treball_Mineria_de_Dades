if (!require(ggplot2)) install.packages("ggplot2")
if (!require(scales)) install.packages("scales")
if (!require(patchwork)) install.packages("patchwork")
if (!require(car)) install.packages("car")
if (!require(glmnet)) install.packages("glmnet")
library(glmnet)
library(car)
library(ggplot2)
library(scales)
library(patchwork)
library(dplyr)

#====================================================================
# S'HA DE CARREGAR TAMBÉ EL RDATA DE IMPUTACIÓ ======================
#====================================================================

# Boxplots 
numeric_vars <- setdiff(names(data_imputed_MICE)[sapply(data, is.numeric)], "Exited")

num_plots <- lapply(numeric_vars, function(var) {
  ggplot(data_imputed_MICE, aes(x = Exited, y = .data[[var]], fill = Exited)) +
    geom_boxplot() +
    labs(title = var, x = NULL, y = NULL) +
    theme_minimal() +
    theme(legend.position = "none",
          plot.title = element_text(hjust = 0.5, size = 10))
})

num_panel <- wrap_plots(num_plots, ncol = 3) +
  plot_annotation(title = "Boxplots de variables numéricas vs Exited")

num_panel


# Kruskal - wallis =========================================

kruskal_results <- lapply(numeric_vars, function(var) {
  # fórmula: variable ~ Exited
  formula <- as.formula(paste("Exited ~ ", var))
  test <- kruskal.test(formula, data = data_imputed_MICE)
  
  data.frame(
    Variable = var,
    Statistic = test$statistic,
    p_value = test$p.value,
    stringsAsFactors = FALSE
  )
})

kruskal_table <- do.call(rbind, kruskal_results)
kruskal_results = as.data.frame(kruskal_results)
kruskal_results %>% select(Variable, p_value) %>% where(p_value > 0.05)

# ============================
# 📊 3️⃣ Ordenar por p-valor
# ============================
kruskal_table <- kruskal_table[order(kruskal_table$p_value), ]

kruskal_table

# Barplots 
cat_vars <- setdiff(names(data_imputed_MICE)[sapply(data_imputed_MICE, is.factor)], "Exited")

cat_plots <- lapply(cat_vars[-4], function(var) {
  ggplot(data_imputed_MICE[,-7], aes(x = .data[[var]], fill = Exited)) +
    geom_bar(position = "fill") +
    scale_y_continuous(labels = percent_format()) +
    labs(title = var, x = NULL, y = NULL) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "none",
          plot.title = element_text(hjust = 0.5, size = 10))
})

cat_panel <- wrap_plots(cat_plots, ncol = 3) +
  plot_annotation(title = "Proporciones de Exited según variables categóricas")

print(cat_panel)

data_imputed_MICE$IsActiveMember = as.factor(data_imputed_MICE$IsActiveMember)

ggplot(data_imputed_MICE, aes(x = IsActiveMember, y = Balance, fill = Exited)) +
    geom_boxplot() +
    labs(title = var, x = NULL, y = NULL) +
    theme_minimal() +
    theme(legend.position = "none",
          plot.title = element_text(hjust = 0.5, size = 10))
# GLM
formula_glm <- as.formula(paste("Exited ~", paste(setdiff(names(data), "Exited"), collapse = " + ")))

modelo_glm <- glm(formula_glm, data = data_imputed_MICE, family = binomial)

summary(modelo_glm)
vif(modelo_glm)


## LASSO (L1)
# Crea matriz de diseño (sin intercept)
x <- model.matrix(Exited ~ ., data = data_imputed_MICE[,-c(7,18)])[, -1]

# Variable respuesta binaria (glmnet la requiere numérica 0/1)
y <- as.numeric(data_imputed_MICE$Exited) - 1  # convierte factor {1,2} a {0,1}

set.seed(123) 

cv_lasso <- cv.glmnet(
  x, y,
  alpha = 1,               # 1 = LASSO
  family = "binomial",     # regresión logística
  nfolds = 10
)

# Lambda óptimo
lambda_opt <- cv_lasso$lambda.min
cat("Lambda óptimo seleccionado:", lambda_opt, "\n")


coef_matrix <- coef(cv_lasso, s = "lambda.min")
lasso_coefs <- as.data.frame(as.matrix(coef_matrix))
colnames(lasso_coefs) <- "Estimate"
lasso_coefs$Variable <- rownames(lasso_coefs)

lasso_coefs <- lasso_coefs[order(-abs(lasso_coefs$Estimate)), ]  # ordenar por magnitud

print(lasso_coefs)

