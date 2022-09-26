# "SVMR" "SVML" "SVMS" "RF" "KNN" "DCT" "LR" "HV" "SV"
#!/bin/bash
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