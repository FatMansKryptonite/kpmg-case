import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay
import shap


def make_roc(model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series, location: str = 'results/') -> None:
    RocCurveDisplay.from_estimator(model, X, y)
    plt.show()
    plt.savefig(f'{location}/roc.pdf', bbox_inches='tight')


def make_shap_plots(model: RandomForestClassifier, X: pd.DataFrame, location: str = 'results/shap') -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    plt.figure()
    shap.plots.beeswarm(shap_values[:, :, 1])
    plt.savefig(f'{location}/beeswarm.pdf', bbox_inches='tight')

    for feature in X.columns:
        plt.figure()
        shap.dependence_plot(feature, shap_values.values[:, :, 1], X)
        plt.savefig(f'{location}/{feature}.pdf', bbox_inches='tight')
