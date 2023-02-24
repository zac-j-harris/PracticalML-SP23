# Zac Harris
# 
# Set and check working Directory
setwd("~/Desktop/  /SP 23/Practical ML/")
getwd()


# Import dataset
white_wine = read.csv("./wine_quality/winequality-white.csv", sep=";")

# Generate pairs graph
# pairs(white_wine)


# Grab dimensions
dim(white_wine)

# Generate 
z_scores <- as.data.frame(sapply(white_wine, function(ww) (abs(ww-mean(ww))/sd(ww))))

no_outliers <- white_wine[!rowSums(z_scores>5), ]
dim(no_outliers)

# Generate pairs graph
# pairs(no_outliers)

# Save the new csv
write.csv(no_outliers, "./wine_quality/fixed-winequality-whites.csv")
