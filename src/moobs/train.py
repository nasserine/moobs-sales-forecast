
def train_model(model, X_train, y_train):
    """
    Entraîne le modèle sur les données d'entraînement.

    Parameters
    ----------
    model : estimateur compatible scikit-learn
        Modèle à entraîner, par exemple XGBRegressor.
    X_train : DataFrame
        Variables explicatives d'entraînement.
    y_train : Series
        Ventes réelles d'entraînement.

    Returns
    -------
    Le modèle entraîné.
    """
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    """
    Produit des prédictions avec un modèle entraîné.

    Parameters
    ----------
    model : estimateur entraîné
        Modèle qui a déjà été entraîné avec train_model().
    X_test : DataFrame
        Variables explicatives des données de test.

    Returns
    -------
    Les ventes prédites.
    """
    return model.predict(X_test)
