# Paquets necessaris MEMD
# -----------------------

install.packages(c(
  #"tidyverse",   # readr, dplyr, ggplot2...
  #"caret",       # ML i preprocessament
  #"data.table",  # grans datasets
  #"skimr",       # resum ràpid de dades
  #"DataExplorer",
  #"psych",
  #"naniar",
  #"dlookr",
  #"missForest",
  #"mi"
))

library(mi)
library(missForest)  
library(tidyverse)
library(caret)
library(data.table)
library(skimr)
library(DataExplorer)
library(psych)
library(naniar)
library(ggplot2)
library(tidyr)
library(dplyr)
library(dlookr)

# Codi
# ----

# Importar data d'entrenament
#====================
data <- read_csv("train.csv", col_types = cols(...1 = col_skip()))

# Convertir les variables que no són números en categòriques
for(i in 1:ncol(data)){
  if(is.character(data[[i]])==TRUE){
    data[[i]] = as.factor(data[[i]])
  }
}

# Convertir les variables que prenen valors numèrics però són categòriques en categòriques
data$HasCrCard <- as.factor(data$HasCrCard)
data$IsActiveMember <- as.factor(data$IsActiveMember)
data$SavingsAccountFlag <- as.factor(data$SavingsAccountFlag)
data$Exited <- as.factor(data$Exited)
#====================


# Obtenir metadata de les dades
#====================
plot_str(data)
#====================


# Mirar si tenim alguna entrada reptida 
#====================
any(duplicated(na.omit(data)$ID)) # Diu que no n'hi ha
# He mirat només l'ID pq com s'han posat NA a totes les entrades potser hi havia alguna repetida però amb NA en diferents llocs
# Tot i així, com hi ha NAs al ID no sé molt bé que pensar
#====================


# Eliminar columnes no necessàries per estudiar les dades
#====================
data <- data %>% select(-ID, -Surname) #ID i Surname
#====================

# Estudi preeliminar de les dades que tenim: quàntes columnes numèriques, categòriques, valors NA,...
#====================
introduce(data) %>% View()
plot_intro(data)
plot_missing(data) # Cada variable té un 30% de NA
#====================

# Estudiar de les entrades amb MOLTS NAs
#====================
# Comptar els NAs de cada fila
na_per_row <- rowSums(is.na(data))

# Obtenir les files amb més de 10 NAs (més de la meitat de les variables)
filas_moltes_na <- which(na_per_row > 10)
length(filas_moltes_na)
#====================


# Separem les variables numèriques i les variables categòriques, per facilitar l'estudi
#====================
clases <- sapply(data, class)
varNum <- names(clases)[which(clases %in% c("numeric", "integer"))]
varCat <- names(clases)[which(clases %in% c("character", "factor"))]
#====================


# Descriu les dades numèriques: mitjana, max, min,...
#====================
psych::describe(data[, varNum])
#====================


# Mira correlacions entre les variables numèriques
#====================
plot_correlation(na.omit(data[, varNum]), maxcat = 15)
# Així podem saber si podem eliminar alguna variable per facilitar l'estudi
#====================


# Diagrama de barres de cada variable categòrica
#====================
for (col in varCat) {
  p <- ggplot(data, aes_string(x = col)) +
    geom_bar(fill = "darkseagreen4") +
    theme_minimal() +
    labs(title = paste("Distribució de", col),
         x = col,
         y = "Comptatge")
  print(p)
}
# Aquí podem veure el desbalanceig en algunes variables
#====================


# Histogrames de les variables numèriques:
#====================
for (col in varNum) {
  p <- ggplot(data, aes_string(x = col)) +
    geom_histogram(fill = "tomato", bins = 30, color = "black") +
    theme_minimal() +
    labs(title = paste("Histograma de", col),
         x = col,
         y = "Freqüència")
  print(p)
}
#====================


# Boxplot per les variables numèriques
#====================
for (col in varNum) {
  p <- ggplot(data, aes_string(y = col)) +
    geom_boxplot(fill = "lightgreen", color = "black", na.rm = TRUE) +
    theme_minimal() +
    labs(
      title = paste("Boxplot de", col),
      y = col
    )
  print(p)
}
#====================

# Missing data
#====================
colSums(is.na(iris))
iris.mis <- missForest::prodNA(iris, noNA = 0.1)
colSums((is.na(iris.mis)))
#iris.mis <- mi::create.missing(iris, pct.mis = 10)
naniar::mcar_test(iris.mis)

aq_shadow <- bind_shadow(airquality)
airquality %>%
  bind_shadow() %>%
  group_by(Ozone_NA) %>%
  summarise_at(.vars = "Solar.R",
               .funs = c("mean", "sd", "var", "min", "max"),
               na.rm = TRUE)

prop_miss_case(airquality)
#================



