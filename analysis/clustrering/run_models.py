from sklearn import linear_model
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from scipy.spatial.distance import squareform
import numpy as np


def run_model(model, embs, name):
    if len(model.shape) == 2: # matrix
        model = squareform(model) # upper triangle list

    y = model.reshape(-1, 1)
    X = np.array(embs)

    kf = KFold(n_splits=5)

    coef_10 = []
    r2_10 = []
    rho_10 = []
    for train_index, test_index in kf.split(X):
        regr = linear_model.Ridge()
        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)

        coef = regr.coef_
        # print("Coefficients: ", coef)

        r2 = r2_score(y_test, y_pred)
        # print("Coefficient of determination: %.2f" % r2)

        rho = stats.spearmanr(y_test, y_pred)[0]
        # print("Spearman rho: %.2f" % rho)

        coef_10.append(coef)
        r2_10.append(r2)
        rho_10.append(rho)

    d = dict([("name", [name]),
             ("coef_mean", np.mean(coef_10, axis=0).tolist() ),
             ("coef_std", np.std(coef_10, axis=0).tolist()),
             ("r2_mean", [np.mean(r2_10)]),
             ("r2_std", [np.std(r2_10)]),
             ("rho_mean", [np.mean(rho_10)]),
             ("rho_std", [np.std(rho_10)])]
            )

    return d