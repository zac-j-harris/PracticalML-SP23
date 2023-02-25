# Zac Harris
# 
# Set and check working Directory
setwd("~/Documents/Github/PracticalML-SP23/Assignment 1/")
getwd()

# Import dataset
white_wine = read.csv("./Data/wine_quality/winequality-white.csv", sep=";")

# Generate pairs graph
pairs(white_wine)

# Print the dimensions
dim(white_wine)

# Generate, function sourced from: https://www.statology.org/remove-outliers-r/
z_scores <- as.data.frame(sapply(white_wine, function(ww) (abs(ww-mean(ww))/sd(ww))))

# Remove all points >5 std away from mean.
no_outliers <- white_wine[!rowSums(z_scores>5), ]

# Print the new dimensions
dim(no_outliers)

# Generate new pairs graph
pairs(no_outliers)

# Save the new csv
write.csv(no_outliers, "./Data/wine_quality/fixed-winequality-whites.csv")
