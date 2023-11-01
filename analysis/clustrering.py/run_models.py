from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from scipy.spatial.distance import squareform


def run_model(model, embs):
    if len(model.shape) == 2: # matrix
        model = squareform(model) # upper triangle list

    y = model.reshape(-1, 1)
    X = embs
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regr = linear_model.Ridge()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)

    # The coefficients
    print("Coefficients: ", regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
    # The rho spearman correlation
    print("Spearman rho: %.2f" % stats.spearmanr(y_test, y_pred)[0])