library(ggplot2)
library(tidyr)
library(caret)
# df is the data frame
df <- read.csv("dataset.csv")
head(df)

df$Column2[df$Column2 == "M"] <- 1 # Changing the M to 1 And B to 0
df$Column2[df$Column2 == "B"] <- 0
df$Column2 <- as.numeric(df$Column2)
head(df)

features <- df[, 3:32]  # Select columns 3 to 32 as features
target <- df[, 2]       # Select column 2 as the target variable
head(features)
head(target)
# Create a new data frame with selected columns
new_df <- df[, c(3:32, 2)]
target_counts <- table(df[, 2])
# Print columns with null values and their counts
str(new_df)
null_counts <- colSums(is.na(new_df))


# Print the counts
cat("Count of 0 That is Benign :", target_counts[1], "\n")
cat("Count of 1 That is Malignant :", target_counts[2], "\n")

# X_train and X_test are your feature matrices
# Scale the training set
# Set seed for reproducibility
set.seed(2)
# Create a random split
split_indices <- createDataPartition(target, p = 0.8, list = FALSE)
# Create training and testing sets
X_train <- features[split_indices, ]
X_test <- features[-split_indices, ]
Y_train <- target[split_indices]
Y_test <- target[-split_indices]

head(X_test)
head(X_train)
head(Y_test)
head(Y_train)


X_train_std <- scale(X_train)

# Standardize the testing set features using the mean and standard deviation from the training set
X_test_std <- scale(X_test, center = attr(X_train_std, "scaled:center"), scale = attr(X_train_std, "scaled:scale"))

# Train a linear regression model
linear_model <- lm(Y_train ~ ., data = data.frame(Y_train, X_train_std))

# Print the summary of the model
summary(linear_model)
print(summary)


#  linear_model is trained linear regression model
# X_test_std is  standardized testing set features
# Y_test is  actual testing set target variable

# Make predictions on the standardized testing set
predictions_std <- predict(linear_model, newdata = data.frame(X_test_std))

# Reverse the standardization to get predictions on the original scale
predictions <- predictions_std * sd(Y_test) + mean(Y_test)

# Evaluate the model
mae <- mean(abs(Y_test - predictions))
rmse <- sqrt(mean((Y_test - predictions)^2))
rsquared <- 1 - (sum((Y_test - predictions)^2) / sum((Y_test - mean(Y_test))^2))

print(paste("Mean Absolute Error (MAE):", mae))
print(paste("Root Mean Squared Error (RMSE):", rmse))
print(paste("R-squared:", rsquared))






# Create a density plot
density_plot <- ggplot(df, aes(x = Column4, y = ..density.., fill = factor(..count..))) +
  geom_density(alpha = 0.7, color = "red", fill = "lightcoral") +
  labs(title = "Density Plot for Column 4 i.e.Texture1",
       x = "Values",
       y = "Density") +
  theme_minimal()
# Print the density  plot
print(density_plot)




# Create a more detailed histogram for the 4th column
histogram_plot <- ggplot(df, aes(x = Column4, fill = ..count..)) +
  geom_histogram(binwidth = 1, color = "black", fill = "skyblue", alpha = 0.7) +
  geom_density(aes(y = ..count.. * 0.005), fill = "orange", alpha = 0.7) +  # Add a density plot
  labs(title = "Detailed Histogram for Column 4, i.e.Texture1",
       x = "Values",
       y = "Frequency") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), 
        legend.position = "none")  # Remove legend

# Print the histogram
print(histogram_plot)

library(ggplot2)
]
selected_columns <- names(df)[4:8]

head(selected_columns)

# Reshape the data frame to long format
df_long <- tidyr::gather(df, key = "Column", value = "Value", selected_columns)

# Set custom column names
new_column_names <- c("Texture1", "Perimeter1", "Area1", "Smoothness1", "Compactness1")

# Update the Column variable in df_long with the new names
df_long$Column <- factor(df_long$Column, levels = unique(df_long$Column), labels = new_column_names)

# Create the detailed violin plot with custom X-axis labels
violin_plot_columns4to8 <- ggplot(df_long, aes(x = Column, y = Value, fill = Column)) +
  geom_violin(scale = "width", draw_quantiles = c(0.25, 0.5, 0.75), trim = FALSE) +
  labs(title = "Detailed Violin Plot for Columns 4-8",
       x = "Columns",
       y = "Values") +
  scale_fill_manual(values = c("skyblue", "salmon", "red", "lightcoral", "lightblue")) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_x_discrete(labels = new_column_names)  # Set custom X-axis labels

# Print thE violin plot
print(violin_plot_columns4to8)











