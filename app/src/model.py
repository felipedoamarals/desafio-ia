from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

@dataclass
class ModelResult:
    clf: RandomForestClassifier
    cm: np.ndarray
    report: str


def train_model(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> ModelResult:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return ModelResult(clf=clf, cm=cm, report=report)