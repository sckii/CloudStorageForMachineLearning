from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys

from utils.inputs import read_arqs
from utils.classifiers import get_default_classifiers, set_classifier
from utils.period import get_all_windows
from utils.results import generate_results

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
sns.set()
plt.rcParams["figure.figsize"] = (12, 6)

def initialize_training_parameters(arq_acessos, random_state=42, **kwargs):
    num_weeks = arq_acessos.shape[1]
    scaler = kwargs.get("scaler", MinMaxScaler(feature_range=(0, 1)))
    resample = kwargs.get("resample", RandomUnderSampler(random_state=random_state))
    clfs_dict = kwargs.get("clfs_dict", get_default_classifiers(True, random_state))
    
    return [
        pd.DataFrame(),
        range((num_weeks // steps_to_take) - ((2 * window_size) // steps_to_take)),
        scaler,
        resample,
        clfs_dict,
    ]

def train_ML(initial_train, end_train, initial_train_label, end_train_label, resample, classifier_name, classifiers_dictionary):
    # TRAINING
    acc_train = arq_acessos.iloc[:, initial_train:end_train]

    # filtrando volume 0 (objetos que ainda não existem nessa janela de tempo)
    vol_train = arq_vol_bytes.iloc[:, end_train - 1]
    idx = vol_train.values.ravel() > 0.0
    acc_train = acc_train[idx]

    # train_label = arq_classes.iloc[:, idx_train_label]
    train_label = arq_classes.iloc[:, initial_train_label:end_train_label]  #
    train_label = train_label.apply(
        lambda row: int(row.any()), axis=1
    )  # Aqui se existir algum 1 na janela de label de treino então pegamos 1, se não, 0.

    train_label = train_label[idx]  # filtrando volume 0

    # dealing with the imbalanced dataset
    # X_train, y_train = resample.fit_resample(acc_train, train_label)
    clf = set_classifier(classifier_name, classifiers_dictionary)
    clf.fit(acc_train, train_label)
    return clf


def evaluate(initial_evaluation, end_evaluation, initial_evaluation_label, end_evaluation_label):
    # EVALUATION
    acc_eval = arq_acessos.iloc[:, initial_evaluation:end_evaluation]

    # filtrando volume 0 (objetos que ainda não existem nessa janela de tempo )
    vol_eval = arq_vol_bytes.iloc[:, end_evaluation - 1]
    idx = vol_eval.values.ravel() > 0.0
    acc_eval = acc_eval[idx]

    # eval_label = arq_classes.iloc[:, idx_eval_label]
    eval_label = arq_classes.iloc[:, initial_evaluation_label:end_evaluation_label]  #
    eval_label = eval_label.apply(lambda row: int(row.any()), axis=1)  #

    eval_label = eval_label[idx]  # filtrando volume 0
    return acc_eval, eval_label  # o retorno são os dados de acesso e se há acesso


def predict(classifier_name, acc_eval, clf, initial_evaluation, end_evaluation):
    if classifier_name != "ONLINE":
        X_eval = acc_eval
        y_hat_by_obj = clf.predict(X_eval)
        y_hat_obj_red = y_hat_by_obj  # .apply(lambda row: int(row.any()), axis=1)
    # calculando métricas
    else:
        y_hat_by_obj = arq_classes.iloc[:, initial_evaluation:end_evaluation]
        y_hat_obj_red = y_hat_by_obj.apply(lambda row: int(row.any()), axis=1)
    return y_hat_obj_red


def run_train_eval(classifier_name, random_state, window_size, steps_to_take, pop, print_df=True, FP_penalty_enabled=True, **kwargs):

    (df_metrics_eval, time_total, scaler, resample,classifiers_dictionary) = initialize_training_parameters(arq_acessos, random_state, **kwargs)

    for time_window in time_total:

        (initial_train, end_train, initial_train_label, end_train_label, initial_evaluation, end_evaluation, initial_evaluation_label, end_evaluation_label) = get_all_windows(time_window, window_size, steps_to_take)

        if classifier_name != "ONLINE":
            clf = train_ML( initial_train, end_train, initial_train_label, end_train_label, resample, classifier_name, classifiers_dictionary)

        acc_eval, eval_label = evaluate(initial_evaluation, end_evaluation, initial_evaluation_label, end_evaluation_label)

        if classifier_name != "ONLINE":
            y_hat_obj_red = predict(classifier_name, acc_eval, clf, initial_evaluation, end_evaluation)
        else:
            y_hat_obj_red = predict(classifier_name, acc_eval, None, initial_evaluation, end_evaluation)

        first_col, last_col = acc_eval.columns[0], acc_eval.columns[-1]

        df = pd.DataFrame()
        df_to_evaluate = pd.DataFrame()
        df["label"] = eval_label.squeeze().values
        df["pred"] = y_hat_obj_red

        vol_eval = arq_vol_bytes.iloc[:, initial_evaluation_label:end_evaluation_label]
        acc_eval = arq_acessos.iloc[:, initial_evaluation_label:end_evaluation_label]
        df["total_vol"] = vol_eval.sum(axis=1)

        acc_with_no_zeroes = acc_eval[df.total_vol > 0]
        vol_with_no_zeroes = vol_eval[df.total_vol > 0]
        df_with_no_zeroes  = df[df.total_vol > 0]

        for i in range(steps_to_take):
            df_to_evaluate["vol_bytes"] = vol_with_no_zeroes.iloc[:, i]
            df_to_evaluate["acc_fut"] = acc_with_no_zeroes.iloc[:, i]
            df_to_evaluate["label"] = df_with_no_zeroes.label[:]
            df_to_evaluate["pred"] = df_with_no_zeroes.pred[:]
            eval_metrics = generate_results(
                df_to_evaluate,
                f"{first_col}:{last_col}",
                pop,
                classifier_name,
                FP_penalty_enabled,
            )
            df_metrics_eval = pd.concat([df_metrics_eval, eval_metrics])

    # if print_df:
    #     display(df_metrics_eval.loc[:, [*h_cols, *costs_cols]])
    #     display(df_metrics_eval.drop(columns=costs_cols))
    return df_metrics_eval


pop = sys.argv[1]
clf_name = sys.argv[2]
window_size = int(sys.argv[3])
steps_to_take = int(sys.argv[4])

# agregar resultados? proximo parametro
# habilitar grafico
print_df = False
# penalidade para falso positivo
FP_penalty_enabled = True  # param de entrada

arq_acessos, arq_classes, arq_vol_bytes = read_arqs(pop)
costs = pd.DataFrame()
results = {}
random_state = 42
progressionBarDescription = None

costs_over = pd.DataFrame()
results_over = {}
costs_smote = pd.DataFrame()
results_smote = {}

results[clf_name] = run_train_eval(clf_name, random_state, window_size, steps_to_take, pop)
costs[clf_name] = results[clf_name][["cost ml", "cost all hot", "cost opt"]].sum()
print(costs)
# endregion
