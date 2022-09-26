from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (VotingClassifier, RandomForestClassifier)

import sys

def get_default_classifiers(probability=False, random_state=42):
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


def set_classifier(clf_key, clfs_dict):
    clf = clfs_dict.get(clf_key, None)
    if clf is None:
        print("Unknown classifier!")
        sys.exit(1)
    return clf