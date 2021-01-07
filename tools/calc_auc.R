#!/usr/bin/env Rscript

library("optparse")
library("pROC")
library("rjson")

opt_parser <- OptionParser(usage="%prog result_json_file")
opt <- parse_args(opt_parser, positional_arguments=1)

content <- fromJSON(file=opt$args)

y_pred_adeno <- c(length(content))
y_pred_squamous <- c(length(content))
y_true_adeno <- c(length(content))
y_true_squamous <- c(length(content))
for (i in 1:length(content)) {
    data = content[[i]]

    y_pred_adeno[[i]] <- data$y_pred[[2]]
    y_pred_squamous[[i]] <- data$y_pred[[3]]
    y_true_adeno[[i]] <- if(data$y_true == 1) 1 else 0
    y_true_squamous[[i]] <- if(data$y_true == 2) 1 else 0
}

auc_adeno <- auc(
    response=y_true_adeno, 
    predictor=y_pred_adeno, 
)
ci_adeno <- ci.auc(
    auc=auc_adeno,
    conf.level=0.95, 
    method="delong",
)
test_adeno <- roc.test(
    response=y_true_adeno,
    predictor1=y_pred_adeno,
    predictor2=rep(c(0.5), times=length(y_pred_adeno)),
    method="delong",
    alternative="two.sided",
)
print("### Adeno ###")
print(auc_adeno)
print(ci_adeno)
print(test_adeno)

auc_squamous <- auc(
    response=y_true_squamous, 
    predictor=y_pred_squamous, 
)
ci_squamous <- ci.auc(
    auc=auc_squamous,
    conf.level=0.95, 
    method="delong",
)
test_squamous <- roc.test(
    response=y_true_squamous,
    predictor1=y_pred_squamous,
    predictor2=rep(c(0.5), times=length(y_pred_squamous)),
    method="delong",
    alternative="two.sided",
)
print("### Squamous ###")
print(auc_squamous)
print(ci_squamous)
print(test_squamous)
