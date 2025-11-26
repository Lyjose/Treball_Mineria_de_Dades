library(pacman)
pacman::p_load(pscl, ggplot2, ROCR, lightgbm, methods, Matrix, caret)

set.seed(123)

load("dataAREG_outliers.RData")
data <- dataAREG

data$DigitalEngagementScore <- factor(data$DigitalEngagementScore)
data$CustomerSegment <- factor(data$CustomerSegment)
data$NetPromoterScore <- factor(data$NetPromoterScore)