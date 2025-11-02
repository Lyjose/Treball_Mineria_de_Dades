library(arules)
library(arulesViz)
library(FactoMineR)
library(tidyverse)



load("dataAREG_outliers.RData")
data <- dataAREG


clases <- sapply(data, class)

varNum <- names(clases)[which(clases %in% c("numeric", "integer"))]
varCat <- names(clases)[which(clases %in% c("character", "factor"))]

# Funcio per categoritzar les númeriques
categorizar_por_percentiles <- function(data, varNum, n_percentiles = 4) {
  data_categorizado <- data
  
  for (var in varNum) {
    # Obtener cortes según los percentiles
    percentiles <- quantile(data[[var]], probs = seq(0, 1, length.out = n_percentiles + 1), na.rm = TRUE, type = 7)
    # Asegurar que los valores repetidos en los cortes no causen error
    percentiles <- unique(percentiles)
    
    # Categorizar usando cut
    data_categorizado[[var]] <- as.factor(
      cut(
      data[[var]],
      breaks = percentiles,
      include.lowest = TRUE,
      labels = FALSE)
    )
  }
  return(data_categorizado)
}

data_categorizado = categorizar_por_percentiles(data,varNum)
str(data_categorizado)

# Pasar les dades a format transaccions
transac <- as(data_categorizado,"transactions")
summary(transac)
class(transac)



# Ahora creamos itemsets

itemsets <- apriori(data = transac,
                    parameter = list(support = 0.05,
                                     minlen = 1,
                                     maxlen = 5,
                                     target = "frequent itemset"))
inspect(itemsets[1:5]) # els 5 primers per veure q tal

# Els 20 primers itemsets amb suport més alt
top_20_itemsets <- sort(itemsets, by = "support", decreasing = TRUE)[1:20]
(inspect(top_20_itemsets))

# Els 20 primers itemsets amb suport més alt i més d'un item
inspect(sort(itemsets[size(itemsets) > 1], decreasing = TRUE)[1:20])

########## RESULTAT IMPORTANT #########
# Els 20 primers itemsets que continguin Exited = 1
itemsets_filtrado <- arules::subset(itemsets,
                                    subset = items %in% "Exited=1")
inspect(itemsets_filtrado[1:20])

# Els 20 primers itemsets que continguin Exited = 0
itemsets_filtrado <- arules::subset(itemsets,
                                    subset = items %in% "Exited=0")
inspect(itemsets_filtrado[1:20])



########## Regles, configurar support, confidence i maxlen #########
rules = apriori (transac, parameter = list (support=0.01, confidence=0.4, maxlen = 5, minlen=2))
#rules = apriori (transac, parameter = list (support=0.01, confidence=0.8, maxlen = 5, minlen=2))
#rules = apriori (transac, parameter = list (support=0.05, confidence=0.8, maxlen = 5, minlen=2))
#rules = apriori (transac, parameter = list (support=0.35, confidence=0.9, maxlen = 5, minlen=2))
summary(rules)


filtrado_reglas <- subset(x = rules,
                          subset = rhs %ain% "Exited=1")
inspect(filtrado_reglas)
filtrado_reglas

# Eliminació de regles redundants
reglas_redundantes <- rules[is.redundant(x = filtrado_reglas, measure = "confidence")]
reglas_Noredund <- rules[!is.redundant(x = filtrado_reglas, measure = "confidence")]
reglas_Noredund <- sort(reglas_Noredund,by="lift")

reglas_Exited_1 = subset(x = reglas_Noredund,
                    subset = rhs %ain% "Exited=1")
reglas_Exited_0 = subset(x = reglas_Noredund,
                    subset = rhs %ain% "Exited=0")

# Resultat important.
inspect(reglas_Exited_0[1:10])
inspect(reglas_Exited_1[1:10])

