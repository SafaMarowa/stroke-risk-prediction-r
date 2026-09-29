#================================================================================
# STROKE RISK PREDICTION - R Script
# Author : Safa Marowa Tasfia
# Dataset: Kaggle Stroke Prediction Dataset (fedesoriano, 2021)
# Method : Follows Joshi & Dhakal's approach to predicting type 2 diabetes
#          using logistic regression and machine learning
#================================================================================
 
# Block 1 - Install and Load Packages
required_packages <- c('leaps', 'rpart', 'rpart.plot')
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}
library(leaps); library(rpart); library(rpart.plot)  # 'tree' package removed: unused, since Blocks 17-18 use rpart()
 
# Block 2 - Load The Kaggle Dataset
stroke_data = read.csv("healthcare-dataset-stroke-data.csv")
head(stroke_data)
nrow(stroke_data)
ncol(stroke_data)
names(stroke_data)
 
# Block 3 - Understand the data
str(stroke_data)
summary(stroke_data)
table(stroke_data$stroke)
stroke_rate <- mean(stroke_data$stroke, na.rm = TRUE) * 100
cat('Stroke rate: ', round(stroke_rate,1), '%\n')
cat('Missing BMI values: ', sum(is.na(stroke_data$bmi)), '\n')
cat('Gender values : ', unique(stroke_data$gender), '\n')
cat('Married values: ', unique(stroke_data$ever_married), '\n')
cat('Residence values: ', unique(stroke_data$Residence_type), '\n')
cat('Smoking values: ', unique(stroke_data$smoking_status), '\n')
cat('Work type values: ', unique(stroke_data$work_type), '\n')
table(stroke_data$stroke)
prop.table(table(stroke_data$stroke))
 
# Block 4 - Clean the data
stroke_data <- stroke_data[,-1]
ncol(stroke_data)
stroke_data$bmi <- as.numeric(stroke_data$bmi)
median_bmi <- median(stroke_data$bmi, na.rm = TRUE)
stroke_data$bmi[is.na(stroke_data$bmi)] <- median_bmi
cat('Missing BMI after fix: ', sum(is.na(stroke_data$bmi)), '\n')
stroke_data$avg_glucose_level <- as.numeric(stroke_data$avg_glucose_level)
stroke_data$age <- as.numeric(stroke_data$age)
stroke_data$gender <- ifelse(as.character(stroke_data$gender) == 'Male', 1, 0)
stroke_data$ever_married <- ifelse(as.character(stroke_data$ever_married) == 'Yes', 1, 0)
stroke_data$Residence_type <- ifelse(as.character(stroke_data$Residence_type) == 'Urban', 1, 0)
stroke_data$smoking_status <- ifelse(
  as.character(stroke_data$smoking_status) == 'smokes', 2, ifelse(
  as.character(stroke_data$smoking_status) == 'formerly smoked', 1, 0))
stroke_data$work_type <- ifelse(
  as.character(stroke_data$work_type) == 'Private', 0, ifelse(
  as.character(stroke_data$work_type) == 'Self-employed', 1, ifelse(
  as.character(stroke_data$work_type) == 'Govt.job', 2, ifelse(
  as.character(stroke_data$work_type) == 'children', 3, 4))))
stroke_data$stroke <- as.numeric(stroke_data$stroke)
cat('Missing values per column after cleaning: ', colSums(is.na(stroke_data)), '\n')
head(stroke_data)
summary(stroke_data)
 
# Block 5 - Descriptive statistics
colMeans(stroke_data)
cat('-----Standard Deviations-----')
cat('SD of age : ', round(sd(stroke_data$age),2), '\n')
cat('SD of glucose level : ', round(sd(stroke_data$avg_glucose_level),2), '\n')
cat('SD of bmi : ', round(sd(stroke_data$bmi),2), '\n')
 
# Block 6 - Model 1: Null Model
model1 <- glm(stroke ~ 1, data = stroke_data, family = binomial)
summary(model1)
AIC(model1)
 
# Block 7 - Model 2: Full Model
model2 <- glm(stroke ~ age + gender + hypertension + heart_disease +
                 ever_married + work_type + Residence_type + avg_glucose_level +
                 bmi + smoking_status,
               data = stroke_data, family = binomial)
summary(model2)
AIC(model2)
BIC(model2)
 
# Block 8 - Model 3: Reduced model
model3 <- glm(stroke ~ age + hypertension + heart_disease +
                 avg_glucose_level + bmi + smoking_status,
               data = stroke_data, family = binomial)
summary(model3)
AIC(model3)
BIC(model3)
 
# Block 9 - Model 4: Add first interaction term
model4 <- glm(stroke ~ age + hypertension + heart_disease + avg_glucose_level +
                 bmi + smoking_status + age:hypertension,
               data = stroke_data, family = binomial)
summary(model4)
AIC(model4)
 
# Block 10 - Model 5: Proposed best model
model5 <- glm(stroke ~ age + hypertension + heart_disease + avg_glucose_level +
                 bmi + smoking_status + age:hypertension + avg_glucose_level:bmi,
               data = stroke_data, family = binomial)
summary(model5)
AIC(model1, model2, model3, model4, model5)
BIC(model1, model2, model3, model4, model5)
 
# Block 11 - Best Subset Selection
best_sub <- regsubsets(
  stroke ~ age + hypertension + heart_disease + avg_glucose_level + bmi +
    smoking_status + gender + ever_married + work_type + Residence_type,
  data = stroke_data, nvmax = 10)
plot(best_sub, scale = 'bic', main = 'BIC: Which variables are most important?')
plot(best_sub, scale = 'adjr2', main = 'Adjusted R2: Which variables are most important?')
plot(best_sub, scale = 'Cp', main = 'Mallows Cp: Which variables are most important?')
 
# Block 12 - Split data into training and test sets
set.seed(42)
train_index <- sample(1:nrow(stroke_data), size = round(0.75 * nrow(stroke_data)))
train_set <- stroke_data[train_index,]
test_set <- stroke_data[-train_index,]
cat('Training set size:', nrow(train_set), '\n')
cat('Test set size: ', nrow(test_set), '\n')
 
# Block 13 - Train model 5 on training data
model5_train <- glm(stroke ~ age + hypertension + heart_disease +
                       avg_glucose_level + bmi + smoking_status +
                       age:hypertension + avg_glucose_level:bmi,
                     data = train_set, family = binomial)
 
# Block 14 - Predict on test set
pred_prob <- predict(model5_train, newdata = test_set, type = 'response')
pred_class <- ifelse(pred_prob > 0.5, 1, 0)
 
# Block 15 - Confusion matrix and accuracy
conf_mat <- table(Actual = test_set$stroke, Predicted = pred_class)
print(conf_mat)
accuracy <- sum(diag(conf_mat)) / sum(conf_mat)
cat('Test Set Accuracy:', round(accuracy*100,2), '%\n')
cat('Test Set Error :', round((1-accuracy)*100,2), '%\n')
 
# Block 15b - ROC-AUC on the test set (rank-based, no extra packages required)
# Reproduces the AUC value reported in the paper without depending on the
# pROC package, using the Mann-Whitney U relationship between AUC and the
# rank-sum statistic.
pos_prob <- pred_prob[test_set$stroke == 1]
neg_prob <- pred_prob[test_set$stroke == 0]
auc_value <- (sum(outer(pos_prob, neg_prob, '>')) + 0.5 * sum(outer(pos_prob, neg_prob, '=='))) /
  (length(pos_prob) * length(neg_prob))
cat('Test ROC-AUC:', round(auc_value, 3), '\n')
 
# Block 15c - Threshold sensitivity analysis
# Shows how accuracy/sensitivity/specificity trade off as the classification
# threshold is lowered below the conventional 0.5 cutoff.
for (th in c(0.5, 0.3, 0.2, 0.15, 0.1, 0.05)) {
  pc <- ifelse(pred_prob > th, 1, 0)
  TP <- sum(pc == 1 & test_set$stroke == 1)
  TN <- sum(pc == 0 & test_set$stroke == 0)
  FP <- sum(pc == 1 & test_set$stroke == 0)
  FN <- sum(pc == 0 & test_set$stroke == 1)
  acc <- (TP + TN) / length(pc)
  sens <- TP / (TP + FN)
  spec <- TN / (TN + FP)
  cat('threshold', th, '-> accuracy', round(acc * 100, 2), '% sensitivity',
      round(sens * 100, 1), '% specificity', round(spec * 100, 1), '%\n')
}
 
# Block 16 - 8-Fold Cross-Validation
set.seed(42)
k <- 8
fold_numbers <- rep(1:k, times = ceiling(nrow(stroke_data)/k))
fold_id <- sample(head(fold_numbers, nrow(stroke_data)))
fold_accuracy <- c()
for (i in 1:k) {
  test_fold <- stroke_data[fold_id == i,]
  train_fold <- stroke_data[fold_id != i,]
  cv_model <- glm(stroke ~ age + hypertension + heart_disease +
                     avg_glucose_level + bmi + smoking_status +
                     age:hypertension + avg_glucose_level:bmi,
                   data = train_fold, family = binomial)
  cv_prob <- predict(cv_model, newdata = test_fold, type = 'response')
  cv_pred <- ifelse(cv_prob > 0.5, 1, 0)
  this_acc <- mean(cv_pred == test_fold$stroke)
  fold_accuracy <- c(fold_accuracy, this_acc)
}
cv_accuracy <- mean(fold_accuracy)
cv_error <- 1 - cv_accuracy
cat('CV Accuracy (8-fold):', round(cv_accuracy*100,2), '%\n')
cat('CV Error (8-fold):', round(cv_error*100,2), '%\n')
 
# Block 17 - Classification tree
library(rpart)
library(rpart.plot)
stroke_data$stroke_cat <- as.factor(ifelse(stroke_data$stroke == 1, "Yes", "No"))
train_tree <- stroke_data[train_index, ]
test_tree <- stroke_data[-train_index, ]
full_tree <- rpart(stroke_cat ~ age + hypertension + heart_disease +
                      avg_glucose_level + bmi + smoking_status,
                    data = train_tree, method = "class",
                    parms = list(loss = matrix(c(0,5,20,0), nrow = 2)),
                    control = rpart.control(cp = 0.0001, minsplit = 10, maxdepth = 6))
print(full_tree)
rpart.plot(full_tree, type = 2, extra = 104, main = "Full Classification Tree - Stroke Risk")
tree_prob <- predict(full_tree, newdata = test_tree, type = "prob")
tree_pred <- ifelse(tree_prob[, "Yes"] > 0.3, "Yes", "No")
tree_pred <- as.factor(tree_pred)
tree_conf <- table(Actual = test_tree$stroke_cat, Predicted = tree_pred)
print(tree_conf)
tree_acc <- sum(diag(tree_conf)) / sum(tree_conf)
cat("Full Tree Accuracy:", round(tree_acc * 100, 2), "%\n")
 
# Block 18 - Prune the tree
# NOTE: with this loss matrix and cp path, prune(cp = 0.0005) does not
# reduce the tree below its unpruned 11-terminal-node size (verified on rerun).
printcp(full_tree)
plotcp(full_tree)
pruned_tree <- prune(full_tree, cp = 0.0005)
rpart.plot(pruned_tree, type = 2, extra = 104, main = "Pruned Classification Tree - Stroke Risk")
pruned_prob <- predict(pruned_tree, newdata = test_tree, type = "prob")
pruned_pred <- ifelse(pruned_prob[, "Yes"] > 0.3, "Yes", "No")
pruned_pred <- as.factor(pruned_pred)
pruned_conf <- table(Actual = test_tree$stroke_cat, Predicted = pruned_pred)
print(pruned_conf)
pruned_acc <- sum(diag(pruned_conf)) / sum(pruned_conf)
cat("Pruned Tree Accuracy:", round(pruned_acc * 100, 2), "%\n")
 
# Block 18b - ROC-AUC for the classification tree (same rank-based method as Block 15b)
tree_pos <- pruned_prob[, "Yes"][test_tree$stroke_cat == "Yes"]
tree_neg <- pruned_prob[, "Yes"][test_tree$stroke_cat == "No"]
tree_auc <- (sum(outer(tree_pos, tree_neg, '>')) + 0.5 * sum(outer(tree_pos, tree_neg, '=='))) /
  (length(tree_pos) * length(tree_neg))
cat('Tree Test ROC-AUC:', round(tree_auc, 3), '\n')
 
# Block 19 - Final Accuracy On All Patients
final_prob <- predict(model5, type = 'response')
final_class <- ifelse(final_prob > 0.5, 1, 0)
final_conf <- table(Actual = stroke_data$stroke, Predicted = final_class)
print(final_conf)
final_acc <- sum(diag(final_conf)) / sum(final_conf)
cat('\n==================================================\n')
cat('FINAL RESULTS SUMMARY\n')
cat('==================================================\n')
cat('Dataset : Kaggle Stroke Dataset\n')
cat('Total patients :', nrow(stroke_data), '\n')
cat('Model 5 Accuracy :', round(final_acc*100,2), '%\n')
cat('CV Error (8-fold):', round(cv_error*100,2), '%\n')
cat('Pruned Tree Acc :', round(pruned_acc*100,2), '%\n')
cat('Key predictors: age, hypertension, heart_disease,\n')
cat(' avg_glucose_level, bmi\n')
cat('==================================================\n')
#================================================================================
# End of Script
#================================================================================

