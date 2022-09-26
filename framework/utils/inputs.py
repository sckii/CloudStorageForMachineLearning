import pandas as pd

# Metodos de entrada
def read_pop():
    pop = input("Digite a base a ser avaliada: \n 1- Pop1 \n 2- Pop2\n")
    while pop != "1" and pop != "2":
        pop = input(
            "Opção inválida. \nDigite a base a ser avaliada: \n 1- Pop1 \n 2- Pop2\n"
        )
    pop = "Pop" + pop
    return pop


def read_FP_penalty():
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

def read_arqs(pop):
    PATH = "../eval/"
    name_space = []
    arq = open(PATH + "nsJanelas_" + pop + ".txt", "r")
    arqPath = lambda name: f"{PATH}/{name}_{pop}.txt"

    for line in arq:
        name_space.append(sorted(list(map(int, line.split()))))

    arq_acessos = pd.read_csv(arqPath("access"), low_memory=False, sep=" ", index_col="NameSpace")
    arq_classes = pd.read_csv(arqPath("target"), low_memory=False, sep=" ", index_col="NameSpace")
    arq_vol_bytes = pd.read_csv(arqPath("vol_bytes"), low_memory=False, sep=" ", index_col="NameSpace")

    return arq_acessos, arq_classes, arq_vol_bytes
