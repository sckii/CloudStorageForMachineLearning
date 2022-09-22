 
# Previsão da Classe de Frequência de Acesso de Objetos em Serviços de Armazenamento em Nuvem

## Dependências
- Pandas
- Matplotlib
- Numpy
- Scikit_learn
- Imblearn
- Imbalanced_learn
## Setando Arquivo de execução
Na pasta framework crie o arquivo de execução, exemplo: "testePop1.sh" para fazer o teste na Pop1
``` 
    #Exmplo do condigo do testePop1.sh
    pop="Pop1"
    mkdir $"results/"${pop}
    declare -a classifiers=("ONLINE" "SVMR" "SVML" "SVMS" "RF" "KNN" "DCT" "LR" "HV" "SV")
    window="4"
    stepstotake=4
    timestamp=$(date +%y%m%d-%H%M)
    folder="results/"${pop}"/"${timestamp}
    mkdir $folder
    for i in "${classifiers[@]}"
    do
        echo "Gerando saída em:" $folder"/"$i".txt"
        python3 framework.py $pop $i $window $stepstotake >> $folder"/"$i".txt"
    done
```
## Comando de execução
Para o exemplo a cima
```
    bash testePop1.sh
```
## TODO
- Melhorar Filtragem dos Dados
- Modificar função de custo
- Mudar esquema de semanas
- Avaliar o periodo de teste/treino
