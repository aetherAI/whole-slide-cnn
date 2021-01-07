#!/usr/bin/env Rscript

library("optparse")
library("pROC")
library("rjson")

opt_parser <- OptionParser(usage="%prog result_json_file_1 result_json_file_2")
opt <- parse_args(opt_parser, positional_arguments=2)

content1 <- fromJSON(file=opt$args[[1]])
content2 <- fromJSON(file=opt$args[[2]])

y_true_adeno1 <- c(length(content1))
y_true_squamous1 <- c(length(content1))
y_pred_adeno_1 <- c(length(content1))
y_pred_squamous_1 <- c(length(content1))
for (i in 1:length(content1)) {
    data = content1[[i]]

    y_true_adeno1[[i]] <- if(data$y_true == 1) 1 else 0
    y_true_squamous1[[i]] <- if(data$y_true == 2) 1 else 0
    y_pred_adeno_1[[i]] <- data$y_pred[[2]]
    y_pred_squamous_1[[i]] <- data$y_pred[[3]]
}

y_true_adeno2 <- c(length(content2))
y_true_squamous2 <- c(length(content2))
y_pred_adeno_2 <- c(length(content2))
y_pred_squamous_2 <- c(length(content2))
for (i in 1:length(content2)) {
    data = content2[[i]]

    y_true_adeno2[[i]] <- if(data$y_true == 1) 1 else 0
    y_true_squamous2[[i]] <- if(data$y_true == 2) 1 else 0
    y_pred_adeno_2[[i]] <- data$y_pred[[2]]
    y_pred_squamous_2[[i]] <- data$y_pred[[3]]
}

roc1_adeno <- roc(
    response=y_true_adeno1,
    predictor=y_pred_adeno_1,
)
roc2_adeno <- roc(
    response=y_true_adeno2,
    predictor=y_pred_adeno_2,
)
test_adeno <- roc.test(
    roc1=roc1_adeno,
    roc2=roc2_adeno,
    method="delong",
)
print("### Adeno ###")
print(test_adeno)

roc1_squamous <- roc(
    response=y_true_squamous1,
    predictor=y_pred_squamous_1,
)
roc2_squamous <- roc(
    response=y_true_squamous2,
    predictor=y_pred_squamous_2,
)
test_squamous <- roc.test(
    roc1=roc1_squamous,
    roc2=roc2_squamous,
    method="delong",
)
print("### Squamous ###")
print(test_squamous)
