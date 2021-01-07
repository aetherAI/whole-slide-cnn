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

print("### Adeno ###")
for (i in 1:100) {
    sample_list = sample.int(
        n=length(content),
        size=length(content),
        replace=TRUE,
    )
    sampled_y_true_adeno = y_true_adeno[sample_list]
    sampled_y_pred_adeno = y_pred_adeno[sample_list]
    auc_adeno <- auc(
        response=sampled_y_true_adeno, 
        predictor=sampled_y_pred_adeno,
        quiet=TRUE,
    )
    cat(i, as.character(as.numeric(auc_adeno)), "\n", sep="\t")
}

print("### Squamous ###")
for (i in 1:100) {
    sample_list = sample.int(
        n=length(content),
        size=length(content),
        replace=TRUE,
    )
    sampled_y_true_squamous = y_true_squamous[sample_list]
    sampled_y_pred_squamous = y_pred_squamous[sample_list]
    auc_squamous <- auc(
        response=sampled_y_true_squamous, 
        predictor=sampled_y_pred_squamous,
        quiet=TRUE,
    )
    cat(i, as.character(as.numeric(auc_squamous)), "\n", sep="\t")
}
