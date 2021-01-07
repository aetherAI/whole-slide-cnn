import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

if __name__ == "__main__":
    ## Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_and_labels",
        type=str,
        nargs="+",
        help="One or more pairs of a testing result json file, a label and a color hex, e.g. test_result.json:MIL:#02183F",
    )
    args = parser.parse_args()
    
    ## Read JSON
    content_list = []
    label_list = []
    color_hex_list = []
    for file_and_label in args.file_and_labels:
        filename, label, color_hex = file_and_label.split(":")
        with open(filename) as f:
            content = json.load(f)
        content_list.append(content)
        label_list.append(label)
        color_hex_list.append(color_hex)
        
    ## Draw ROC
    for target_id, target in [(1, "adeno"), (2, "squamous")]:
        plt.figure(figsize=(5.25, 4.5))
        for i in range(len(content_list)):
            y_true_list = np.array([row["y_true"] == target_id for row in content_list[i]])
            y_pred_list = np.array([row["y_pred"][target_id] for row in content_list[i]])

            fpr, tpr, _ = sklearn.metrics.roc_curve(
                y_true=y_true_list,
                y_score=y_pred_list,
            )
            auc = sklearn.metrics.roc_auc_score(
                y_true_list,
                y_pred_list,
            )
            plt.plot(
                fpr,
                tpr,
                color=color_hex_list[i],
                lw=2.0,
                label="{} (AUC={:.4f})".format(label_list[i], auc),
            )

        plt.plot([0, 1], [0, 1], color='#595959', lw=2.0, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.savefig('{}.pdf'.format(target))
        print("ROC curve figure saved to {}.pdf .".format(target))
