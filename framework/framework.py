from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    VotingClassifier,
    StackingClassifier,
    RandomForestClassifier,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import (
    precision_recall_curve,
    f1_score,
    fbeta_score,
    auc,
    make_scorer,
    confusion_matrix,
)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import defaultdict, namedtuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys
import seaborn as sns
from tqdm import tqdm_notebook

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
sns.set()
plt.rcParams["figure.figsize"] = (12, 6)

# region constants
Hot, Warm = [0, 1]
CostStorageHot, CostStorageWarm = [0.0230, 0.0125]
CostOperationHot, CostOperationWarm = [0.0004, 0.0010]
CostRetrievalHot, CostRetrievalWarm = [0.0000, 0.0100]
# endregion

# region Metodos de entrada
def readPop():
    pop = input("Digite a base a ser avaliada: \n 1- Pop1 \n 2- Pop2\n")
    while pop != "1" and pop != "2":
        pop = input(
            "Opção inválida. \nDigite a base a ser avaliada: \n 1- Pop1 \n 2- Pop2\n"
        )
    pop = "Pop" + pop
    return pop


def readFPPenalty():
    FPPenalty = input(
        "Habilitar penalidade aos falsos positivos: \n 1- Sim \n 2- Não\n"
    )
    while FPPenalty != "1" and FPPenalty != "2":
        FPPenalty = input(
            "Opção inválida. \Habilitar penalidade aos falsos positivos: \n 1- Sim \n 2- Não\n"
        )
    if FPPenalty == "1":
        FPPenalty = True
    else:
        FPPenalty = False
    return FPPenalty


def readArqs(pop):
    PATH = "../eval/"
    name_space = []
    arq = open(PATH + "nsJanelas_" + pop + ".txt", "r")
    for line in arq:
        name_space.append(sorted(list(map(int, line.split()))))
    arqPath = lambda name: f"{PATH}/{name}_{pop}.txt"
    arq_acessos = pd.read_csv(
        arqPath("access"), low_memory=False, sep=" ", index_col="NameSpace"
    )
    arq_classes = pd.read_csv(
        arqPath("target"), low_memory=False, sep=" ", index_col="NameSpace"
    )
    arq_vol_bytes = pd.read_csv(
        arqPath("vol_bytes"), low_memory=False, sep=" ", index_col="NameSpace"
    )
    return arq_acessos, arq_classes, arq_vol_bytes


# endregion


def objectCost(vol_gb, acc, obj_class):
    acc_1k = float(acc) / 1000.0  # acc prop to 1000
    cost = 0
    if obj_class == Hot:
        cost = (
            vol_gb * CostStorageHot
            + acc_1k * CostOperationHot
            + vol_gb * acc * CostRetrievalHot
        )
    else:  # warm
        cost = (
            vol_gb * CostStorageWarm
            + acc_1k * CostOperationWarm
            + vol_gb * acc * CostRetrievalWarm
        )
    # print(f"obj_class = {obj_class} | vol_gb = {vol_gb} | acc = {acc} | cost = {cost}") #DEBUG
    return cost


def thresholdAccess(vol_gb, obj_class):
    if obj_class == "HW":  # hot to warm
        return int(
            vol_gb
            * (CostStorageWarm - CostStorageHot)
            / (
                CostOperationHot
                - CostOperationWarm
                - vol_gb * 1000 * (CostRetrievalWarm - CostRetrievalHot)
            )
        )


def getOptimalCost(acc_fut, vol_gb, costs):
    # Limiares de acesso para camadas H-W e W-C
    acc_thres_hw = thresholdAccess(vol_gb, "HW")
    QQ0 = QW0 = 0
    if acc_fut > acc_thres_hw:  # HOT
        costs["opt"] += objectCost(vol_gb, acc_fut, Hot)
        QQ0 += 1
    else:  # WARM
        costs["opt"] += objectCost(vol_gb, acc_fut, Warm)
        QW0 += 1


def getClassifierCost(row, costs, vol_gb, acc_fut, FPPenaltyEnabled):
    if row["label"] == 0 and row["pred"] == 0:
        # Se o objeto é warm e modelo acertou, adiciona custo de mudança pra warm
        costs["TN"] += objectCost(vol_gb, acc_fut, Warm)
    elif row["label"] == 0 and row["pred"] == 1:
        # Se o objeto é warm e modelo errou, adiciona penalidade por erro
        # if FPPenaltyEnabled:
        costs["FP"] += objectCost(vol_gb, acc_fut, Warm)  # - penalty
        costs["FP"] += objectCost(vol_gb, acc_fut, Hot)  # - accesses in hot tier
    elif row["label"] == 1 and row["pred"] == 0:
        # Se o objeto é hot e modelo errou, adiciona penalidade por erro
        costs["FN"] += objectCost(vol_gb, 1, Warm)  # - one access to return to hot tier
        # costs["FN"] += objectCost(vol_gb, acc_fut - 1, Hot)  # - accesses in hot tier
    elif row["label"] == 1 and row["pred"] == 1:
        # Se o objeto é hot e modelo acertou, adiciona custo de permanencia em hot
        costs["TP"] += objectCost(vol_gb, acc_fut, Hot)


def costsOfAllClassifiers(costs):
    pred_cost = costs["TP"] + costs["FN"] + costs["TN"] + costs["FP"]  # Custo1
    opt_cost = costs["opt"]
    default_cost = costs["always_H"]  #: always hot
    default_rcs = (default_cost - opt_cost) / default_cost
    pred_rcs = (default_cost - pred_cost) / default_cost
    return {
        "rcs ml": pred_rcs,
        "rcs opt": default_rcs,
        "cost ml": pred_cost,
        "cost opt": opt_cost,
        "cost all hot": default_cost,
    }


def calculateCosts(df, single_score=False, FPPenaltyEnabled=True):
    costs = defaultdict(float)  # default = 0

    for index, row in df.iterrows():
        vol_gb = float(row["vol_bytes"]) / (1024.0**3)  # vol per GB
        acc_fut = row["acc_fut"]

        # Custo otimo
        # costs vai por referencia, volta com os valores preenchidos nas colunas
        getOptimalCost(acc_fut, vol_gb, costs)

        # Custo classificador
        getClassifierCost(row, costs, vol_gb, acc_fut, FPPenaltyEnabled)

        # Custo simples sem otimização
        costs["always_H"] += objectCost(vol_gb, acc_fut, Hot)  #: always hot

    pred_cost = costs["TP"] + costs["FN"] + costs["TN"] + costs["FP"]  # Custo1
    if single_score:
        return pred_cost
    else:
        return costsOfAllClassifiers(costs)


def calculateMetrics(df, window=0, rel=False):
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


def generateResults(df, window, log_id, clf_name, FPPenaltyEnabled):
    metrics = calculateMetrics(df)
    costs = calculateCosts(df)

    results = {
        "Dados": log_id,
        "Model": clf_name,
        #                "Time window": window,
        "Qtd obj": df.shape[0],
        **metrics.to_dict(orient="list"),
        **costs,
    }
    return pd.DataFrame(results, index=[window])


def aggregateResults(df_results, gp_objs=5, std=False):
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


def getDefaultClassifiers(probability=False, random_state=42):
    default_clfs = {
        "SVMR": svm.SVC(
            gamma="auto", probability=probability, random_state=random_state
        ),  # rbf
        "SVML": svm.SVC(
            kernel="linear", probability=probability, random_state=random_state
        ),
        "SVMS": svm.SVC(
            kernel="sigmoid", probability=probability, random_state=random_state
        ),
        "RF": RandomForestClassifier(n_jobs=-1, random_state=random_state),
        "KNN": KNeighborsClassifier(n_jobs=-1),
        "DCT": DecisionTreeClassifier(random_state=random_state),
        "LR": LogisticRegression(n_jobs=-1),
    }
    return {
        **default_clfs,
        "HV": VotingClassifier(list(default_clfs.items()), voting="hard", n_jobs=-1),
        "SV": VotingClassifier(list(default_clfs.items()), voting="soft", n_jobs=-1),
    }


def setClassifier(clf_key, clfs_dict):
    clf = clfs_dict.get(clf_key, None)
    if clf is None:
        print("Unknown classifier!")
        sys.exit(1)
    return clf


def initializeTrainingParameters(arq_acessos, random_state=42, **kwargs):
    num_weeks = arq_acessos.shape[1]
    scaler = kwargs.get("scaler", MinMaxScaler(feature_range=(0, 1)))
    resample = kwargs.get("resample", RandomUnderSampler(random_state=random_state))
    clfs_dict = kwargs.get("clfs_dict", getDefaultClassifiers(True, random_state))
    return [
        pd.DataFrame(),
        range((num_weeks // stepsToTake) - ((2 * windowSize) // stepsToTake)),
        scaler,
        resample,
        clfs_dict,
    ]


def getAllWindows(time_window, windowSize, stepsToTake):
    steps = stepsToTake * time_window
    initialTrain, endTrain, initialTrainLabel, endTrainLabel = getPeriodStamps(
        steps, windowSize
    )
    (
        initialEvaluation,
        endEvaluation,
        initialEvaluationLabel,
        endEvaluationLabel,
    ) = getPeriodStamps(initialTrainLabel, windowSize)
    endEvaluationLabel = initialEvaluationLabel + stepsToTake
    return [
        initialTrain,
        endTrain,
        initialTrainLabel,
        endTrainLabel,
        initialEvaluation,
        endEvaluation,
        initialEvaluationLabel,
        endEvaluationLabel,
    ]


def getPeriodStamps(time_window, windowSize):
    firstPeriod, lastPeriod = getPeriodByWindow(time_window, windowSize)
    firstPeriodLAbel, lastPeriodLabel = getPeriodByWindow(lastPeriod, windowSize)
    return [firstPeriod, lastPeriod, firstPeriodLAbel, lastPeriodLabel]


def getPeriodByWindow(time_window, windowSize):
    first_period_week = time_window
    last_period_week = first_period_week + windowSize
    return [first_period_week, last_period_week]


def trainML(
    initialTrain,
    endTrain,
    initialTrainLabel,
    endTrainLabel,
    resample,
    classifierName,
    classifiersDictionary,
):
    # TRAINING
    acc_train = arq_acessos.iloc[:, initialTrain:endTrain]

    # filtrando volume 0 (objetos que ainda não existem nessa janela de tempo)
    vol_train = arq_vol_bytes.iloc[:, endTrain - 1]
    idx = vol_train.values.ravel() > 0.0
    acc_train = acc_train[idx]

    # train_label = arq_classes.iloc[:, idx_train_label]
    train_label = arq_classes.iloc[:, initialTrainLabel:endTrainLabel]  #
    train_label = train_label.apply(
        lambda row: int(row.any()), axis=1
    )  # Aqui se existir algum 1 na janela de label de treino então pegamos 1, se não, 0.

    train_label = train_label[idx]  # filtrando volume 0

    # dealing with the imbalanced dataset
    # X_train, y_train = resample.fit_resample(acc_train, train_label)
    clf = setClassifier(classifierName, classifiersDictionary)
    clf.fit(acc_train, train_label)
    return clf


def evaluate(
    initialEvaluation, endEvaluation, initialEvaluationLabel, endEvaluationLabel
):
    # EVALUATION
    acc_eval = arq_acessos.iloc[:, initialEvaluation:endEvaluation]

    # filtrando volume 0 (objetos que ainda não existem nessa janela de tempo )
    vol_eval = arq_vol_bytes.iloc[:, endEvaluation - 1]
    idx = vol_eval.values.ravel() > 0.0
    acc_eval = acc_eval[idx]

    # eval_label = arq_classes.iloc[:, idx_eval_label]
    eval_label = arq_classes.iloc[:, initialEvaluationLabel:endEvaluationLabel]  #
    eval_label = eval_label.apply(lambda row: int(row.any()), axis=1)  #

    eval_label = eval_label[idx]  # filtrando volume 0
    return acc_eval, eval_label  # o retorno são os dados de acesso e se há acesso


def predict(classifierName, acc_eval, clf, initialEvaluation, endEvaluation):
    if classifierName != "ONLINE":
        X_eval = acc_eval
        y_hat_by_obj = clf.predict(X_eval)
        y_hat_obj_red = y_hat_by_obj  # .apply(lambda row: int(row.any()), axis=1)
    # calculando métricas
    else:
        y_hat_by_obj = arq_classes.iloc[:, initialEvaluation:endEvaluation]
        y_hat_obj_red = y_hat_by_obj.apply(lambda row: int(row.any()), axis=1)
    return y_hat_obj_red


def run_train_eval(
    classifierName,
    randomState,
    windowSize,
    stepsToTake,
    pop,
    printDf=True,
    FPPenaltyEnabled=True,
    **kwargs,
):

    (
        df_metrics_eval,
        time_total,
        scaler,
        resample,
        classifiersDictionary,
    ) = initializeTrainingParameters(arq_acessos, randomState, **kwargs)

    for time_window in time_total:

        (
            initialTrain,
            endTrain,
            initialTrainLabel,
            endTrainLabel,
            initialEvaluation,
            endEvaluation,
            initialEvaluationLabel,
            endEvaluationLabel,
        ) = getAllWindows(time_window, windowSize, stepsToTake)

        if classifierName != "ONLINE":
            clf = trainML(
                initialTrain,
                endTrain,
                initialTrainLabel,
                endTrainLabel,
                resample,
                classifierName,
                classifiersDictionary,
            )
        acc_eval, eval_label = evaluate(
            initialEvaluation, endEvaluation, initialEvaluationLabel, endEvaluationLabel
        )
        if classifierName != "ONLINE":
            y_hat_obj_red = predict(
                classifierName, acc_eval, clf, initialEvaluation, endEvaluation
            )
        else:
            y_hat_obj_red = predict(
                classifierName, acc_eval, None, initialEvaluation, endEvaluation
            )

        first_col, last_col = acc_eval.columns[0], acc_eval.columns[-1]

        df = pd.DataFrame()
        dfToEvaluate = pd.DataFrame()
        df["label"] = eval_label.squeeze().values
        df["pred"] = y_hat_obj_red

        vol_eval = arq_vol_bytes.iloc[:, initialEvaluationLabel:endEvaluationLabel]
        acc_eval = arq_acessos.iloc[:, initialEvaluationLabel:endEvaluationLabel]
        df["total_vol"] = vol_eval.sum(axis=1)

        accWithNoZeroes = acc_eval[df.total_vol > 0]
        volWithNoZeroes = vol_eval[df.total_vol > 0]
        dfWithNoZeroes  = df[df.total_vol > 0]

        for i in range(stepsToTake):
            dfToEvaluate["vol_bytes"] = volWithNoZeroes.iloc[:, i]
            dfToEvaluate["acc_fut"] = accWithNoZeroes.iloc[:, i]
            dfToEvaluate["label"] = dfWithNoZeroes.label[:]
            dfToEvaluate["pred"] = dfWithNoZeroes.pred[:]
            eval_metrics = generateResults(
                dfToEvaluate,
                f"{first_col}:{last_col}",
                pop,
                classifierName,
                FPPenaltyEnabled,
            )
            df_metrics_eval = pd.concat([df_metrics_eval, eval_metrics])

    # if printDf:
    #     display(df_metrics_eval.loc[:, [*h_cols, *costs_cols]])
    #     display(df_metrics_eval.drop(columns=costs_cols))
    return df_metrics_eval


# pop = sys.argv[1]
# clf_name = sys.argv[2]
# windowSize = int(sys.argv[3])
# stepsToTake = int(sys.argv[4])
pop = "Pop4"
clf_name = "RF"
windowSize = 4
stepsToTake = 4
# agregar resultados? proximo parametro
# habilitar grafico
printDf = False
# penalidade para falso positivo
FPPenaltyEnabled = True  # param de entrada

arq_acessos, arq_classes, arq_vol_bytes = readArqs(pop)
costs = pd.DataFrame()
results = {}
randomState = 42
progressionBarDescription = None

costs_over = pd.DataFrame()
results_over = {}
costs_smote = pd.DataFrame()
results_smote = {}

results[clf_name] = run_train_eval(clf_name, randomState, windowSize, stepsToTake, pop)

costs[clf_name] = results[clf_name][["cost ml", "cost all hot", "cost opt"]].sum()
print(costs)
# endregion
