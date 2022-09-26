from collections import defaultdict, namedtuple

# region constants
HOT, WARM = [0, 1]
COSTSTORAGEHOT, COSTSTORAGEWARM = [0.0230, 0.0125]
COSTOPERATIONHOT, COSTOPERATIONWARM = [0.0004, 0.0010]
COSTRETRIEVALHOT, COSTRETRIEVALWARM = [0.0000, 0.0100]
# endregion

def object_cost(vol_gb, acc, obj_class):
    acc_1k = float(acc) / 1000.0  # acc prop to 1000
    cost = 0
    if obj_class == HOT:
        cost = (vol_gb * COSTSTORAGEHOT + acc_1k * COSTOPERATIONHOT + vol_gb * acc * COSTRETRIEVALHOT)
    else:  # warm
        cost = (vol_gb * COSTSTORAGEWARM + acc_1k * COSTOPERATIONWARM + vol_gb * acc * COSTRETRIEVALWARM)
    return cost


def threshold_access(vol_gb, obj_class):
    if obj_class == "HW":  # hot to warm
        return int(vol_gb * (COSTSTORAGEWARM - COSTSTORAGEHOT) / ( COSTOPERATIONHOT - COSTOPERATIONWARM - vol_gb * 1000 * (COSTRETRIEVALWARM - COSTRETRIEVALHOT)))


def get_optimal_cost(acc_fut, vol_gb, costs):
    # Limiares de acesso para camadas H-W e W-C
    acc_thres_hw = threshold_access(vol_gb, "HW")
    QQ0 = QW0 = 0
    if acc_fut > acc_thres_hw:  # HOT
        costs["opt"] += object_cost(vol_gb, acc_fut, HOT)
        QQ0 += 1
    else:  # WARM
        costs["opt"] += object_cost(vol_gb, acc_fut, WARM)
        QW0 += 1


def get_classifier_cost(row, costs, vol_gb, acc_fut, FPPenaltyEnabled):
    if row["label"] == 0 and row["pred"] == 0:
        # Se o objeto é warm e modelo acertou, adiciona custo de mudança pra warm
        costs["TN"] += object_cost(vol_gb, acc_fut, WARM)
    elif row["label"] == 0 and row["pred"] == 1:
        # Se o objeto é warm e modelo errou, adiciona penalidade por erro
        # if FPPenaltyEnabled:
        costs["FP"] += object_cost(vol_gb, acc_fut, WARM)  # - penalty
        costs["FP"] += object_cost(vol_gb, acc_fut, HOT)  # - accesses in hot tier
    elif row["label"] == 1 and row["pred"] == 0:
        # Se o objeto é hot e modelo errou, adiciona penalidade por erro
        costs["FN"] += object_cost(vol_gb, 1, WARM)  # - one access to return to hot tier
        # costs["FN"] += object_cost(vol_gb, acc_fut - 1, Hot)  # - accesses in hot tier
    elif row["label"] == 1 and row["pred"] == 1:
        # Se o objeto é hot e modelo acertou, adiciona custo de permanencia em hot
        costs["TP"] += object_cost(vol_gb, acc_fut, HOT)


def costs_of_all_classifiers(costs):
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


def calculate_costs(df, single_score=False, FPPenaltyEnabled=True):
    costs = defaultdict(float)  # default = 0

    for index, row in df.iterrows():
        vol_gb = float(row["vol_bytes"]) / (1024.0**3)  # vol per GB
        acc_fut = row["acc_fut"]

        # Custo otimo
        # costs vai por referencia, volta com os valores preenchidos nas colunas
        get_optimal_cost(acc_fut, vol_gb, costs)

        # Custo classificador
        get_classifier_cost(row, costs, vol_gb, acc_fut, FPPenaltyEnabled)

        # Custo simples sem otimização
        costs["always_H"] += object_cost(vol_gb, acc_fut, HOT)  #: always hot

    pred_cost = costs["TP"] + costs["FN"] + costs["TN"] + costs["FP"]  # Custo1
    if single_score:
        return pred_cost
    else:
        return costs_of_all_classifiers(costs)
