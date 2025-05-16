# Load necessary libraries
library(caret)
library(pROC)
library(randomForest)
library(pROC)
library(ggplot2)

# Load the data
da <- read.csv("depression.csv")

# Convert factors as specified
da$mdeSI <- factor(da$mdeSI)
da$income <- factor(da$income)
da$gender <- factor(da$gender, levels = c("Male", "Female"))
da$age <- factor(da$age)
da$race <- factor(da$race, levels = c("White", "Hispanic", "Black", "Asian/NHPIs", "Other"))
da$insurance <- factor(da$insurance, levels = c("Yes", "No"))
da$siblingU18 <- factor(da$siblingU18, levels = c("Yes", "No"))
da$fatherInHH <- factor(da$fatherInHH)
da$motherInHH <- factor(da$motherInHH)
da$parentInv <- factor(da$parentInv)
da$schoolExp <- factor(da$schoolExp, levels = c("good school experiences", "bad school experiences"))

# Splitting data into training and test sets
set.seed(2027)
trainIndex <- createDataPartition(da$mdeSI, p = 0.75, list = FALSE)
train <- da[trainIndex, ]
test <- da[-trainIndex, ]

#############################################################################################################
#A . LOGISTIC REGRESSION MODEL

#Create model and summary
model <- glm(mdeSI ~ gender + age + race + siblingU18 + parentInv + schoolExp, 
             data = train, family = binomial())
summary(model)
anova(model)

# Calculate exponentiated coefficients (odds ratios) and their confidence intervals
odds_ratios <- exp(coef(model))
conf_intervals <- exp(confint(model, level = 0.95))

# Combine odds ratios with their confidence intervals
result <- data.frame(
  Predictor_Variable = names(odds_ratios),
  AOR = odds_ratios,
  Lower_Bound = conf_intervals[, 1],
  Upper_Bound = conf_intervals[, 2]
)
print(result)

# Make predictions on the train set
predictions <- predict(model, train, type = "response")
predicted_class <- ifelse(predictions > 0.5, "Yes", "No")

# Confusion Matrix, Accuracy, Recall Rate for train data
confusionMatrix <- table(Predicted = predicted_class, Actual = train$mdeSI)
print(confusionMatrix)
accuracy <- sum(diag(confusionMatrix)) / sum(confusionMatrix)
cat("Accuracy:", accuracy, "\n")
recall <- confusionMatrix[2, 2] / sum(confusionMatrix[2, ])  # Sensitivity
cat("Recall (Sensitivity):", recall, "\n")
roc_obj <- roc(as.numeric(train$mdeSI == "Yes"), predictions)
plot(roc_obj, main = "ROC Curve Logistic - Train")
auc_value <- auc(roc_obj)
cat("AUC Score:", auc_value, "\n")

# Make predictions on the test set
predictions <- predict(model, test, type = "response")
predicted_class <- ifelse(predictions > 0.4, "Yes", "No")

# Confusion Matrix, Accuracy, Recall Rate for test data
confusionMatrix <- table(Predicted = predicted_class, Actual = test$mdeSI)
print(confusionMatrix)
accuracy <- sum(diag(confusionMatrix)) / sum(confusionMatrix)
cat("Accuracy:", accuracy, "\n")
recall <- confusionMatrix[2, 2] / sum(confusionMatrix[2, ])  # Sensitivity
cat("Recall (Sensitivity):", recall, "\n")
roc_obj <- roc(as.numeric(test$mdeSI == "Yes"), predictions)
plot(roc_obj, main = "ROC Curve Logistic - Test")
auc_value <- auc(roc_obj)
cat("AUC Score:", auc_value, "\n")


# Plotting Accuracy VS Recall rate for Logistic Model test data
thresholds <- seq(0, 1, by = 0.01)
metrics <- sapply(thresholds, function(t) {
  predicted_class <- ifelse(predictions > t, "Yes", "No")
  # Ensuring both levels are always present in the predicted_class
  predicted_class <- factor(predicted_class, levels = c("Yes", "No"))
  cm <- table(Predicted = predicted_class, Actual = factor(test$mdeSI, levels = c("Yes", "No")))
  accuracy <- sum(diag(cm)) / sum(cm)
  
  # Safely calculate recall, ensuring there are no subscript out of bounds errors
  recall <- if ("Yes" %in% rownames(cm) && "Yes" %in% colnames(cm)) {
    cm["Yes", "Yes"] / sum(cm[, "Yes"])
  } else {
    0  # Return 0 recall if "Yes" class is not present in actual or predicted
  }
  
  c(Accuracy = accuracy, Recall = recall)
})
# Convert to a data frame for plotting
metrics_df <- as.data.frame(t(metrics))
metrics_df$Threshold <- thresholds
# Plot accuracy and recall
ggplot(metrics_df, aes(x = Threshold)) +
  geom_line(aes(y = Accuracy, colour = "Accuracy")) +
  geom_line(aes(y = Recall, colour = "Recall")) +
  labs(title = "Accuracy and Recall vs. Threshold for Logistic Model", x = "Threshold", y = "Rate") +
  scale_colour_manual("", breaks = c("Accuracy", "Recall"), values = c("blue", "red"))

################################################################################################

#RANDOM FOREST MODEL

# Create rf model
rf_model <- randomForest(mdeSI ~ gender + age + race + siblingU18 + parentInv + schoolExp,
                         data = train, ntree = 600, importance = TRUE)

# Variable importance plot with custom palette
varImpPlot(rf_model, main = "Random Forest Variable Importance",
           col = c("#D95F02", "#1B9E77"), pch = 16)

# Define the range of number of trees to evaluate
tree_counts <- seq(100, 1000, by = 100)
oob_errors <- numeric(length(tree_counts))

# Loop through different ntree values
for (i in seq_along(tree_counts)) {
  model <- randomForest(mdeSI ~ gender + age + race + siblingU18 + parentInv + schoolExp,
                        data = train, ntree = tree_counts[i])
  oob_errors[i] <- model$err.rate[tree_counts[i], "OOB"]
}

# Create a data frame for plotting
oob_df <- data.frame(
  Trees = tree_counts,
  OOB_Error = oob_errors
)

# Plot the OOB Error Rate vs Number of Trees
ggplot(oob_df, aes(x = Trees, y = OOB_Error)) +
  geom_line(color = "red") +
  geom_point(color = "red") +
  labs(title = "OOB Error Rate vs. Number of Trees in Random Forest",
       x = "Number of Trees",
       y = "OOB Error Rate") +
  theme_minimal(base_size = 14)


# Make predictions on the train set
predictions_prob <- predict(rf_model, train, type = "prob")[,2]  # probabilities for class "Yes"
predictions_class <- predict(rf_model, train)  # class predictions

# Confusion Matrix, Accuracy, Recall Rate for train data
confusionMatrix <- table(Predicted = predictions_class, Actual = train$mdeSI)
print(confusionMatrix)
accuracy <- sum(diag(confusionMatrix)) / sum(confusionMatrix)
cat("Accuracy:", accuracy, "\n")
recall <- confusionMatrix[2, 2] / sum(confusionMatrix[2, ])
cat("Recall (Sensitivity):", recall, "\n")
roc_obj <- roc(train$mdeSI, predictions_prob)
plot(roc_obj, main = "ROC for Random Forest Model - Train")
auc_value <- auc(roc_obj)
cat("AUC Score:", auc_value, "\n")

# Make predictions on the test set
predictions_prob <- predict(rf_model, test, type = "prob")[,2]  # probabilities for class "Yes"
predictions_class <- predict(rf_model, test)  # class predictions

# Confusion Matrix, Accuracy, Recall Rate for test data
confusionMatrix <- table(Predicted = predictions_class, Actual = test$mdeSI)
print(confusionMatrix)
accuracy <- sum(diag(confusionMatrix)) / sum(confusionMatrix)
cat("Accuracy:", accuracy, "\n")
recall <- confusionMatrix[2, 2] / sum(confusionMatrix[2, ])
cat("Recall (Sensitivity):", recall, "\n")
roc_obj <- roc(test$mdeSI, predictions_prob)
plot(roc_obj, main = "ROC for Random Forest Model - Test")
auc_value <- auc(roc_obj)
cat("AUC Score:", auc_value, "\n")


# plotting accuracy vs recall for RF Model
# Calculate metrics for various thresholds
thresholds <- seq(0, 1, by = 0.01)
metrics <- sapply(thresholds, function(t) {
  predicted_class <- ifelse(predictions_prob > t, "Yes", "No")
  cm <- table(Predicted = factor(predicted_class, levels = c("Yes", "No")), 
              Actual = factor(test$mdeSI, levels = c("Yes", "No")))
  accuracy <- sum(diag(cm)) / sum(cm)
  recall <- cm["Yes", "Yes"] / sum(cm[, "Yes"])
  c(Accuracy = accuracy, Recall = recall)
})
# Convert to a data frame for plotting
metrics_df <- as.data.frame(t(metrics))
metrics_df$Threshold <- thresholds
# Plot accuracy vs. recall
ggplot(metrics_df, aes(x = Threshold)) +
  geom_line(aes(y = Accuracy, colour = "Accuracy")) +
  geom_line(aes(y = Recall, colour = "Recall")) +
  labs(title = "Accuracy and Recall vs. Threshold for Random Forest Model", x = "Threshold", y = "Rate") +
  scale_colour_manual(values = c("Accuracy" = "blue", "Recall" = "red")) +
  theme_minimal()

###########################################################################################################