from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import (precision_recall_curve, f1_score, fbeta_score, auc, make_scorer, confusion_matrix)
import pandas as pd
import numpy as np

from utils.costs import calculate_costs

def calculate_metrics(df, window=0, rel=False):
    y_true, y_pred = df["label"].values, df["pred"].values
    # precision1, recall1, _ = precision_recall_curve(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_samples = len(y_true)

    return pd.DataFrame(
        data={
            "Qtd obj": total_samples,
            "Accuracy": accuracy_score(y_true, y_pred),
            "auc_roc": roc_auc_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "f_beta_2": fbeta_score(y_true, y_pred, beta=2),
            #         "auc": auc(recall1, precision1),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "P": f"{fn + tp} ({(fn + tp) / total_samples:.2%})" if rel else (fn + tp),
            "N": f"{fp + tn} ({(fp + tn) / total_samples:.2%})" if rel else fp + tn,
            "TP": f"{tp} ({tp / total_samples:.2%})" if rel else tp,
            "FP": f"{fp} ({fp / total_samples:.2%})" if rel else fp,
            "FN": f"{fn} ({fn / total_samples:.2%})" if rel else fn,
            "TN": f"{tn} ({tn / total_samples:.2%})" if rel else tn,
        },
        index=[window],
    )


def generate_results(df, window, log_id, clf_name, FPPenaltyEnabled):
    metrics = calculate_metrics(df)
    costs = calculate_costs(df)

    results = {
        "Dados": log_id,
        "Model": clf_name,
        #                "Time window": window,
        "Qtd obj": df.shape[0],
        **metrics.to_dict(orient="list"),
        **costs,
    }
    return pd.DataFrame(results, index=[window])


def aggregate_results(df_results, gp_objs=5, std=False):
    df_mean = pd.DataFrame()

    for i, df in enumerate(df_results[:gp_objs], start=1):
        df = df.drop(columns=["P", "N", "Qtd obj"])  # ,"TP", "FP", "FN", "TN"])
        mean = df.mean()

        if std:
            df_std = df.std()
            df_std.loc[["TP", "FP", "FN", "TN"]] = np.nan

            df = pd.DataFrame(
                data=np.array(
                    [
                        f"{x:.5f} ({y:.2%})" if not np.isnan(y) else f"{x:.2f}"
                        for x, y in zip(mean, df_std)
                    ]
                ).reshape(1, -1),
                columns=df.columns,
                index=[i],
            )
        else:
            df = pd.DataFrame(
                data=np.array([f"{x:.5f}" for x in mean]).reshape(1, -1),
                index=[i],
                columns=df.columns,
            )

        df_mean = pd.concat([df_mean, df]).loc[:, [*df.columns]]

    return df_mean
