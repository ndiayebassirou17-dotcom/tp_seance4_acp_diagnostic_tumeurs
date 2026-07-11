import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    import urllib.request
    import zipfile
    import os

    return (
        ColumnTransformer,
        IsolationForest,
        KNNImputer,
        MinMaxScaler,
        Pipeline,
        RobustScaler,
        SimpleImputer,
        StandardScaler,
        mo,
        np,
        os,
        pd,
        plt,
        stats,
        train_test_split,
        urllib,
        zipfile,
    )


@app.cell
def _(mo):
    mo.md("""
    # TD/TP — Dataset Automobile (UCI ML Repository)
    ## Cours : Collecte, Nettoyage, Préparation et Echantillonnage

    Contexte métier : vous etes data analyst dans une entreprise d'assurance automobile.
    Mission : nettoyer, préparer et analyser les caractéristiques de 205 véhicules (1985)
    pour modeliser le risque d'assurance (symboling).
    """)
    return


@app.cell
def _(os, urllib, zipfile):
    DATA_FILE = "/home/bachir/Documents/data_analyste/datasets/raw/automobile/imports-85.data"
    DATA_DIR = os.path.join("datasets", "raw")
    DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)
    ZIP_URL = "https://archive.ics.uci.edu/static/public/10/automobile.zip"
    ZIP_PATH = os.path.join(DATA_DIR, "automobile.zip")

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        try:
            urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
            with zipfile.ZipFile(ZIP_PATH, "r") as z:
                z.extractall(DATA_DIR)
            if os.path.exists(ZIP_PATH):
                os.remove(ZIP_PATH)
            statut = f"Telecharge et extrait avec succes -> {DATA_PATH}"
        except Exception as e:
            statut = f"Echec du telechargement : {e}. Placez manuellement 'imports-85.data' dans datasets/raw/"
    else:
        statut = f"Fichier deja present : {DATA_PATH}"

    print(statut)
    return (DATA_PATH,)


@app.cell
def _(DATA_PATH, pd):
    COLUMNS = [
        "symboling", "normalized-losses", "make", "fuel-type", "aspiration",
        "num-of-doors", "body-style", "drive-wheels", "engine-location",
        "wheel-base", "length", "width", "height", "curb-weight",
        "engine-type", "num-of-cylinders", "engine-size", "fuel-system",
        "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm",
        "city-mpg", "highway-mpg", "price"
    ]

    df = pd.read_csv(
        DATA_PATH,
        names=COLUMNS,
        na_values=["?"],
        sep=","
    )
    df
    return (df,)


@app.cell
def _(mo):
    mo.md("""
    ## Exercice 1 — Audit du dataset
    """)
    return


@app.cell
def _(df, mo):
    n_rows, n_cols = df.shape
    mo.md(f"1.1 — Dimensions du dataset : {n_rows} lignes et {n_cols} colonnes.")
    return


@app.cell
def _(df):
    df.dtypes
    return


@app.cell
def _(mo):
    mo.md("""
    1.2 — Incohérences de types observees :

    Les colonnes 'horsepower', 'peak-rpm', 'bore', 'stroke' et 'price' devraient etre
    numériques (float), mais elles sont chargées en 'object' si les valeurs '?' ne sont
    pas remplacées par NaN avant le chargement. Avec na_values=['?'], pandas les convertit
    correctement en float. La colonne 'normalized-losses' est aussi en float alors qu'elle
    pourrait sembler entiére. La colonne 'num-of-cylinders' est une chaine de caractéres
    ('four', 'six', etc.) alors qu'elle représente un entier — c'est une incohérence
    notable a corriger pour la modélisation.
    """)
    return


@app.cell
def _(df, pd):
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().mean() * 100).round(2)
    missing_df = pd.DataFrame({
        "manquants": missing_count,
        "pourcentage": missing_pct
    }).sort_values("pourcentage", ascending=False)
    missing_df[missing_df["manquants"] > 0]
    return


@app.cell
def _(mo):
    mo.md("""
    1.3 — Les 3 colonnes les plus touchées par les valeurs manquantes sont
    'normalized-losses' (~20%), 'bore' et 'stroke' (~2%), puis 'horsepower', 'peak-rpm'
    et 'price' (~1-2%). 'normalized-losses' est de loin la plus problématique car elle
    représente la variable cible actuarielle — son imputation doit etre rigoureuse.
    """)
    return


@app.cell
def _(df, mo):
    n_dup = df.duplicated().sum()
    mo.md(f"1.4 — Nombre de doublons stricts (toutes colonnes identiques) : {n_dup}.")
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(mo):
    mo.md("""
    1.5 — Observation sur 'compression-ratio' : la colonne présente un min autour de 7
    et un max autour de 23. Cette plage inhabituelle s'explique par une bimodalite :
    les moteurs essence ont un taux de compression autour de 7-10, tandis que les moteurs
    diesel ont un taux autour de 20-23. Ce sont deux populations physiquement distinctes
    — les traiter comme une seule distribution continue serait une erreur d'analyse.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Exercice 2 — Mecanismes de donnees manquantes
    """)
    return


@app.cell
def _(df):
    nl_by_make = df.groupby("make")["normalized-losses"].apply(
        lambda x: x.isnull().mean() * 100
    ).round(1).sort_values(ascending=False)
    nl_by_make
    return


@app.cell
def _(mo):
    mo.md("""
    2.1 — Le taux de manquants dans 'normalized-losses' varie fortement selon la marque :
    certaines marques n'ont aucune valeur observée, d'autres en ont toutes. Ce pattern
    non homogéne indique que le manquant dépend d'une variable observée ('make') et non
    du hasard pur — c'est un mécanisme MAR (Missing At Random). Le calcul actuariel des
    pertes normalisées est realisé marque par marque, donc les marques avec peu de
    sinistres n'ont pas de valeur calculable. Stratégie recommandée : imputation par
    la mediane par marque.
    """)
    return


@app.cell
def _(df):
    price_missing_by_make = df[df["price"].isnull()]["make"].value_counts()
    price_missing_by_body = df[df["price"].isnull()]["body-style"].value_counts()
    price_missing_by_make, price_missing_by_body
    return


@app.cell
def _(mo):
    mo.md("""
    2.2 — Les valeurs manquantes dans 'price' se concentrent sur quelques marques et
    types de carrosserie spécifiques. Ce pattern suggére un mécanisme MAR : le prix
    manque davantage pour certaines marques dont le catalogue n'était pas disponible
    en 1985. Le manquant depend d'une variable observee ('make'), pas du prix lui-meme.
    Strategie : imputation par la mediane par marque, avec création d'une indicatrice
    'price_imputed'.
    """)
    return


@app.cell
def _(df):
    cooc = df[["bore", "stroke", "horsepower", "peak-rpm"]].isnull().corr()
    cooc
    return


@app.cell
def _(mo):
    mo.md("""
    2.3 — La matrice de co-occurrence des manquants montre si les colonnes 'bore',
    'stroke', 'horsepower' et 'peak-rpm' tendent a manquer ensemble. Une corrélation
    élevée signifie que quand l'une manque, l'autre aussi — généralement lie au fait
    que ces données techniques proviennent de la meme source (fiche constructeur).
    Si la corrélation est forte, un KNNImputer utilisant les colonnes voisines sera
    plus efficace qu'une imputation par médiane simple.

    Tableau recapitulatif des mecanismes :

    | Colonne | Mecanisme | Justification | Strategie |
    |---|---|---|---|
    | normalized-losses | MAR | Depend de 'make' | Mediane par marque |
    | num-of-doors | MCAR | 2 valeurs, oubli administratif | Mode global |
    | bore | MAR | Depend du moteur | KNNImputer ou mediane |
    | stroke | MAR | Meme source que bore | KNNImputer ou mediane |
    | horsepower | MAR | Depend engine-size et engine-type | KNNImputer |
    | peak-rpm | MAR | Lie a horsepower et engine-type | KNNImputer |
    | price | MAR | Depend de 'make' et 'body-style' | Mediane par marque + indicatrice |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Exercice 3 — Strategies d'imputation
    """)
    return


@app.cell
def _(df, plt):
    df_imputed = df.copy()

    nl_before = df_imputed["normalized-losses"].dropna().values

    df_imputed["normalized-losses"] = df_imputed.groupby("make")["normalized-losses"].transform(
        lambda x: x.fillna(x.median())
    )
    mediane_globale_nl = df["normalized-losses"].median()
    df_imputed["normalized-losses"] = df_imputed["normalized-losses"].fillna(mediane_globale_nl)

    nl_after = df_imputed["normalized-losses"].values

    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 4))
    axes1[0].hist(nl_before, bins=20, color="steelblue")
    axes1[0].set_title("normalized-losses avant imputation")
    axes1[1].hist(nl_after, bins=20, color="darkorange")
    axes1[1].set_title("normalized-losses apres imputation")
    plt.tight_layout()
    fig1
    return (df_imputed,)


@app.cell
def _(mo):
    mo.md("""
    3.1 — L'imputation de 'normalized-losses' par la mediane par marque préserve la
    structure du dataset. Pour les marques sans aucune valeur observée, la mediane
    globale est utilisée en repli. L'histogramme apres imputation montre une distribution
    plus compléte dont la forme generale est preservee — signe d'une imputation de qualité.
    """)
    return


@app.cell
def _(df_imputed, mo):
    mode_doors = df_imputed["num-of-doors"].mode()[0]
    df_imputed["num-of-doors"] = df_imputed["num-of-doors"].fillna(mode_doors)
    restant_doors = df_imputed["num-of-doors"].isnull().sum()
    mo.md(f"3.2 — Mode de 'num-of-doors' : '{mode_doors}'. Valeurs manquantes restantes : {restant_doors}.")
    return


@app.cell
def _(KNNImputer, SimpleImputer, df_imputed, mo):
    cols_hp = ["horsepower", "peak-rpm", "engine-size"]

    imp_median = SimpleImputer(strategy="median")
    hp_median = imp_median.fit_transform(df_imputed[cols_hp])

    imp_knn = KNNImputer(n_neighbors=5)
    hp_knn = imp_knn.fit_transform(df_imputed[cols_hp])

    idx_missing_hp = df_imputed["horsepower"].isnull()
    idx_missing_rpm = df_imputed["peak-rpm"].isnull()

    df_imputed["horsepower"] = hp_knn[:, 0]
    df_imputed["peak-rpm"] = hp_knn[:, 1]

    n_hp = int(idx_missing_hp.sum())
    n_rpm = int(idx_missing_rpm.sum())

    mo.md(f"""
    3.3 — Comparaison SimpleImputer vs KNNImputer pour 'horsepower' ({n_hp} valeurs) et 'peak-rpm' ({n_rpm} valeurs) :

    La médiane globale (SimpleImputer) ignore la relation entre puissance et cylindrée.
    Le KNNImputer, en utilisant les 5 voisins les plus proches sur ['horsepower', 'peak-rpm',
    'engine-size'], tient compte de ces correlations et produit des estimations plus
    réalistes. On retient le KNNImputer pour la cohérence physique.
    """)
    return


@app.cell
def _(df_imputed, mo):
    df_imputed["price_imputed"] = df_imputed["price"].isnull().astype(int)

    df_imputed["price"] = df_imputed.groupby("make")["price"].transform(
        lambda x: x.fillna(x.median())
    )
    mediane_globale_price = df_imputed["price"].median()
    df_imputed["price"] = df_imputed["price"].fillna(mediane_globale_price)

    mo.md("""
    3.4 — L'indicatrice 'price_imputed' (1 si la valeur etait manquante, 0 sinon) est
    utile pour la modélisation car elle capture le fait que le prix était inconnu —
    ce qui peut etre un signal prédictif. En l'incluant comme feature, le modèle peut
    apprendre cet effet sans confondre une valeur imputée avec une valeur réelle.
    """)
    return


@app.cell
def _(df_imputed, mo):
    df_imputed["bore"] = df_imputed["bore"].fillna(df_imputed["bore"].median())
    df_imputed["stroke"] = df_imputed["stroke"].fillna(df_imputed["stroke"].median())
    total_final = int(df_imputed.isnull().drop(columns=["price_imputed"]).sum().sum())
    mo.md(f"3.5 — Verification finale : valeurs manquantes restantes = {total_final}. Toutes les colonnes cles ont ete imputees.")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Exercice 4 — Detection et traitement des outliers
    """)
    return


@app.cell
def _(df_imputed, pd):
    def iqr_outliers(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mask = (series < lower) | (series > upper)
        return pd.Series({
            "Q1": round(Q1, 2),
            "Q3": round(Q3, 2),
            "IQR": round(IQR, 2),
            "borne_inf": round(lower, 2),
            "borne_sup": round(upper, 2),
            "n_outliers": int(mask.sum())
        })

    cols_out = ["price", "horsepower", "normalized-losses"]
    iqr_results = pd.DataFrame({c: iqr_outliers(df_imputed[c]) for c in cols_out}).T
    iqr_results
    return (cols_out,)


@app.cell
def _(cols_out, df_imputed, mo, np, pd, stats):
    def zscore_outliers(series):
        z = np.abs(stats.zscore(series.dropna()))
        return int((z > 3).sum())

    z_results = pd.Series(
        {c: zscore_outliers(df_imputed[c]) for c in cols_out},
        name="n_outliers_zscore"
    )
    mo.md(f"""
    4.2 — Comparaison IQR vs Z-score :

    {z_results.to_string()}

    L'IQR détecte généralement plus d'outliers que le Z-score sur des distributions
    asymétriques comme 'price'. Le Z-score est sensible a la moyenne et l'écart-type
    qui sont eux-memes tires par les valeurs extremes, ce qui dilate l'échelle et
    reduit artificiellement le nombre de points detectes. L'IQR, base sur les
    quantilés, est robuste a cette asymétrie.
    """)
    return


@app.cell
def _(IsolationForest, df_imputed, mo):
    cols_iso = ["price", "horsepower", "engine-size", "curb-weight", "normalized-losses"]
    iso = IsolationForest(contamination=0.05, random_state=42)
    df_imputed["iso_label"] = iso.fit_predict(df_imputed[cols_iso])
    n_iso_outliers = int((df_imputed["iso_label"] == -1).sum())
    mo.md(f"4.3 — Isolation Forest (contamination=0.05) detecte {n_iso_outliers} outliers multivaries sur les 5 colonnes numeriques selectionnees.")
    return


@app.cell
def _(df_imputed, mo):
    Q1_p = df_imputed["price"].quantile(0.25)
    Q3_p = df_imputed["price"].quantile(0.75)
    IQR_p = Q3_p - Q1_p
    price_outliers = df_imputed[df_imputed["price"] > Q3_p + 1.5 * IQR_p][["make", "price"]].sort_values("price", ascending=False)
    marques_outliers = price_outliers["make"].value_counts()
    mo.md(f"""
    4.4 — Analyse des outliers de 'price' :

    Les outliers de prix correspondent principalement aux marques de luxe et sportives
    (Mercedes-Benz, BMW, Jaguar, Porsche). Ce sont des valeurs parfaitement légitimes —
    un prix élevé reflete un positionnement premium, pas une erreur de saisie.
    Stratégie adoptée : conservation. Supprimer ces v2hicules biaiserait le modele
    en excluant les voitures haut de gamme qui ont un profil de risque distinctif.
    On peut envisager une transformation log(price) pour reduire l'asymétrie.

    Marques concernees :
    {marques_outliers.to_string()}
    """)
    return IQR_p, Q3_p


@app.cell
def _(IQR_p, Q3_p, df_imputed, plt):
    mask_out = df_imputed["price"] > Q3_p + 1.5 * IQR_p
    df_no_out = df_imputed[~mask_out]

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    axes2[0, 0].boxplot(df_imputed["price"].dropna())
    axes2[0, 0].set_title("Boxplot price (avec outliers)")
    axes2[0, 1].boxplot(df_no_out["price"].dropna())
    axes2[0, 1].set_title("Boxplot price (sans outliers IQR)")
    axes2[1, 0].hist(df_imputed["price"].dropna(), bins=25, color="steelblue")
    axes2[1, 0].set_title("Histogramme price (avec outliers)")
    axes2[1, 1].hist(df_no_out["price"].dropna(), bins=25, color="darkorange")
    axes2[1, 1].set_title("Histogramme price (sans outliers IQR)")
    plt.tight_layout()
    fig2
    return


@app.cell
def _(mo):
    mo.md("""
    ## Exercice 5 — Normalisation et Standardisation
    """)
    return


@app.cell
def _(df_imputed):
    num_cols = [
        "price", "horsepower", "engine-size", "curb-weight", "normalized-losses",
        "city-mpg", "highway-mpg", "wheel-base", "length", "width", "height"
    ]
    df_imputed[num_cols].agg(["min", "max", "std"]).T
    return (num_cols,)


@app.cell
def _(mo):
    mo.md("""
    5.1 — Les plages de valeurs sont tres hétérogènes : 'price' va de ~5000 a ~45000,
    'height' de ~47 a ~60 pouces, 'horsepower' de ~50 a ~300. Sans mise a l'échelle,
    les algorithmes bases sur des distances (KNN, SVM, regression reguliee) seraient
    domines par les colonnes a grande variance, biaisant les resultats. La normalisation
    est indispensable avant toute modélisation.
    """)
    return


@app.cell
def _(MinMaxScaler, RobustScaler, StandardScaler, df_imputed, plt):
    price_vals = df_imputed[["price"]].dropna()

    mm = MinMaxScaler().fit_transform(price_vals)
    ss = StandardScaler().fit_transform(price_vals)
    rb = RobustScaler().fit_transform(price_vals)

    fig3, axes3 = plt.subplots(1, 3, figsize=(13, 4))
    axes3[0].hist(mm, bins=20, color="steelblue")
    axes3[0].set_title("MinMaxScaler [0,1]")
    axes3[1].hist(ss, bins=20, color="seagreen")
    axes3[1].set_title("StandardScaler (mean=0)")
    axes3[2].hist(rb, bins=20, color="darkorange")
    axes3[2].set_title("RobustScaler (mediane/IQR)")
    plt.tight_layout()
    fig3
    return


@app.cell
def _(mo):
    mo.md("""
    5.2 — Les trois scalers produisent des distributions différentes :
    MinMaxScaler compresse tout dans [0,1] mais reste sensible aux outliers.
    StandardScaler centre et reduit mais les outliers restent presents et éloignés.
    RobustScaler utilise la mediane et l'IQR — il est insensible aux valeurs extremes.

    5.3 — Pour ce dataset avec des outliers légitimes (voitures de luxe), le RobustScaler
    est recommandé. Il ne pénalise pas les véhicules premium tout en permettant un
    scaling cohérent pour les algorithmes de ML.
    """)
    return


@app.cell
def _(RobustScaler, df_imputed, num_cols, pd):
    rb_scaler = RobustScaler()
    df_scaled_vals = rb_scaler.fit_transform(df_imputed[num_cols])
    df_scaled = pd.DataFrame(df_scaled_vals, columns=num_cols, index=df_imputed.index)
    df_scaled.median().round(3)
    return (df_scaled,)


@app.cell
def _(mo):
    mo.md("""
    5.4 — Après application du RobustScaler, la médiane de chaque colonne est
    approximativement 0 (par definition du scaler). Une valeur exactement à la médiane
    originale est transformée en 0.
    """)
    return


@app.cell
def _(df_imputed, df_scaled, mo):
    corr_avant = df_imputed["city-mpg"].corr(df_imputed["horsepower"])
    corr_apres = df_scaled["city-mpg"].corr(df_scaled["horsepower"])
    mo.md(f"""
    5.5 — Corrélation de Pearson entre 'city-mpg' et 'horsepower' :
    - Avant scaling : {corr_avant:.4f}
    - Après RobustScaler : {corr_apres:.4f}

    La corrélation lineaire est parfaitement préservée après un scaling monotone.
    Ces transformations sont des fonctions affines qui ne modifient pas les relations
    linéaires entre variables. La corrélation négative confirme la logique physique :
    plus un véhicule est puissant, moins il est économe en carburant.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Exercice 6 — Pipeline complet (ColumnTransformer)
    """)
    return


@app.cell
def _():
    num_features = [
        "price", "horsepower", "engine-size", "curb-weight", "normalized-losses",
        "city-mpg", "highway-mpg", "wheel-base", "length", "width", "height"
    ]
    cat_features = [
        "fuel-type", "aspiration", "body-style", "drive-wheels", "num-of-doors"
    ]
    num_features, cat_features
    return cat_features, num_features


@app.cell
def _(
    ColumnTransformer,
    Pipeline,
    RobustScaler,
    SimpleImputer,
    cat_features,
    num_features,
):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])
    preprocessor
    return (preprocessor,)


@app.cell
def _(cat_features, df, num_features, preprocessor, train_test_split):
    X = df[num_features + cat_features]

    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    X_train_prep.shape, X_test_prep.shape
    return X_test_prep, X_train_prep


@app.cell
def _(X_test_prep, X_train_prep, mo, pd):
    n_missing_train = int(pd.DataFrame(X_train_prep).isnull().sum().sum())
    n_missing_test = int(pd.DataFrame(X_test_prep).isnull().sum().sum())
    mo.md(f"""
    6.3 — Split 80/20 effectue avec random_state=42. Pipeline fitte sur le train uniquement.

    6.4 — Sauvegarde : pd.DataFrame(X_train_prep).to_csv('train_prepared.csv', index=False)

    6.5 — Vérification :
    - Valeurs manquantes train : {n_missing_train}
    - Valeurs manquantes test : {n_missing_test}
    - Dimensions train : {X_train_prep.shape}
    - Dimensions test : {X_test_prep.shape}

    On ne doit jamais fitter le pipeline sur le test set car cela provoquerait une
    fuite de données (data leakage) : le modèle verrait des informations du test
    (mediane, écart-type, modalites) et ses performances en évaluation seraient
    surestimees par rapport a ses performances réelles en production.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Exercice 7 — Echantillonnage
    """)
    return


@app.cell
def _(df, mo):
    srs = df.sample(n=50, replace=False, random_state=42)

    distrib_pop = df["symboling"].value_counts(normalize=True).sort_index()
    distrib_srs = srs["symboling"].value_counts(normalize=True).sort_index()

    couverture_fuel_srs = srs["fuel-type"].value_counts(normalize=True) * 100
    couverture_fuel_pop = df["fuel-type"].value_counts(normalize=True) * 100

    mo.md(f"""
    7.1 — Echantillonnage aleatoire simple (n=50) :

    Distribution symboling population : {distrib_pop.to_dict()}
    Distribution symboling echantillon : {distrib_srs.to_dict()}

    Couverture fuel-type population : {couverture_fuel_pop.round(1).to_dict()}
    Couverture fuel-type SRS : {couverture_fuel_srs.round(1).to_dict()}

    L'échantillon peut ne pas etre representatif car avec n=50 sur 205 lignes,
    les catégories minoritaires (dièsel ~10% = environ 20 vehicules) peuvent etre
    sous- ou sur-échantillonnées par simple variabilité aléatoire. Le SRS ne garantit
    pas la proportionnalit& des strates.
    """)
    return couverture_fuel_pop, couverture_fuel_srs


@app.cell
def _(couverture_fuel_pop, couverture_fuel_srs, df, mo, train_test_split):
    df_clean_fuel = df.dropna(subset=["fuel-type"])
    _, strat_sample = train_test_split(
        df_clean_fuel,
        test_size=50 / len(df_clean_fuel),
        stratify=df_clean_fuel["fuel-type"],
        random_state=42
    )

    couverture_fuel_strat = strat_sample["fuel-type"].value_counts(normalize=True) * 100

    mo.md(f"""
    7.2 — Echantillonnage stratifie sur fuel-type (n~50) :

    Population : {couverture_fuel_pop.round(1).to_dict()}
    SRS : {couverture_fuel_srs.round(1).to_dict()}
    Stratifie : {couverture_fuel_strat.round(1).to_dict()}

    L'échantillonnage stratifie garantit que chaque strate est représentée
    proportionnellement. Il est preferable au SRS quand on s'interesse a des
    sous-groupes minoritaires ou quand la variable d'interet est corrélée avec
    la variable de stratification — ce qui est le cas ici (les dièsels ont des
    profils de risque différents des essences).
    """)
    return


@app.cell
def _(df, mo, np):
    marques = df["make"].dropna().unique()
    np.random.seed(42)
    marques_choisies = np.random.choice(marques, size=5, replace=False)

    cluster_sample = df[df["make"].isin(marques_choisies)]
    n_cluster = len(cluster_sample)
    n_marques = len(marques)

    couverture_body = cluster_sample["body-style"].value_counts(normalize=True) * 100
    couverture_body_pop = df["body-style"].value_counts(normalize=True) * 100

    mo.md(f"""
    7.3 — Echantillonnage par grappes (marques = grappes) :

    Nombre de marques distinctes : {n_marques}
    Marques tirees : {list(marques_choisies)}
    Véhicules obtenus : {n_cluster}

    Couverture body-style population : {couverture_body_pop.round(1).to_dict()}
    Couverture body-style échantillon : {couverture_body.round(1).to_dict()}

    Inconvenient majeur : les véhicules au sein d'une meme marque se ressemblent
    (homogénèité intra-grappe) — l'échantillon peut manquer de diversité si les
    marques selectionnées sont atypiques. La taille de l'échantillon n'est pas
    controlée non plus (elle depend du nombre de véhicules par marque).
    """)
    return


@app.cell
def _(df, mo):
    df_price_ok = df.dropna(subset=["price"])
    df_rich = df_price_ok[df_price_ok["price"] > 20000]
    pct_rich = len(df_rich) / len(df_price_ok) * 100

    sym_risque_pop = (df["symboling"] >= 2).mean() * 100
    sym_risque_rich = (df_rich["symboling"] >= 2).mean() * 100

    mo.md(f"""
    7.4 — Biais d'échantillonnage (price > 20 000 $) :

    Pourcentage du dataset concerne : {pct_rich:.1f}%
    Taux symboling >= 2 dans la population : {sym_risque_pop:.1f}%
    Taux symboling >= 2 dans l'échantillon biaise : {sym_risque_rich:.1f}%

    Ce type de biais s'appelle biais de sélection : l'enqueteur ne capture qu'une
    sous-population non représentative (les véhicules chers). Les véhicules abordables
    sont totalement exclus, ce qui fausse toute conclusion sur la distribution du
    risque d'assurance dans la population génèrale.
    """)
    return


@app.cell
def _(df, mo):
    distrib_sym = df["symboling"].value_counts().sort_index()
    classe_maj = int(distrib_sym.idxmax())
    pct_maj = distrib_sym.max() / len(df) * 100

    mo.md(f"""
    7.5 — Classes desequilibrees pour symboling :

    {distrib_sym.to_string()}

    Classe majoritaire : {classe_maj} ({pct_maj:.1f}% du dataset)

    Deux strategies sans SMOTE :
    1. Sous-echantillonnage (undersampling) : reduire aleatoirement les classes
       majoritaires pour les ramener au niveau de la classe minoritaire. Simple
       mais entraine une perte d'information.
    2. Ponderation des classes (class_weight='balanced') : attribuer un poids
       inversement proportionnel a la frequence de chaque classe dans l'algorithme.
       Le modele penalise davantage les erreurs sur les classes rares sans
       supprimer de donnees.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Exercice 8 — Discussion et Synthese
    """)
    return


@app.cell
def _(df, mo):
    cylinders_map = {
        "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "eight": 8, "twelve": 12
    }
    df_test_cyl = df["num-of-cylinders"].map(cylinders_map)
    valeurs_uniques = sorted(df["num-of-cylinders"].dropna().unique())

    mo.md(f"""
    8.1 — Conversion de 'num-of-cylinders' (textuel -> numerique) :

    Valeurs uniques présentes : {valeurs_uniques}

    Approche 1 — Mapping manuel :
    cylinders_map = {{'two': 2, 'three': 3, 'four': 4, ...}}
    df['num-of-cylinders'].map(cylinders_map)
    Simple, explicite, sans ambiguite. Recommandée quand le vocabulaire est figé.

    Approche 2 — OrdinalEncoder de scikit-learn avec les catégories ordonnées
    explicitement : OrdinalEncoder(catégories=[['two','three','four','five','six',
    'eight','twelve']]).fit_transform(...)
    Intégrable directement dans un Pipeline.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    8.2 — Normalisation de 'compression-ratio' avec distribution bimodale :

    StandardScaler est inapproprie car il suppose une distribution unimodale symztrique
    — la moyenne serait entre les deux pics (autour de 14) alors qu'aucun véhicule n'a
    ce ratio, ce qui rendrait le scaling sans sens physique.

    RobustScaler est plus robuste mais souffre du meme problème fondamental.

    Meilleure approche : creer une variable binaire 'is_dièsel' (1 si fuel-type=dièsel,
    0 sinon) et normaliser séparement les deux sous-populations. Alternativement,
    utiliser 'compression-ratio' comme feature catégorielle discretisée (bas/haut)
    ou l'inclure avec 'fuel-type' dans une intéraction. La transformation log est
    inutile ici car le problème n'est pas l'asymétrie mais la bimodalite.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    8.3 — Colonnes a exclure pour predire symboling :

    | Colonne | Exclure ? | Raison |
    |---|---|---|
    | normalized-losses | Avec prudence | Correlee a symboling, risque de leakage si non disponible en temps réel |
    | price | Non | Feature légitime, connu a la souscription |
    | make | Non (avec encodage) | Informatif sur le profil de risque |
    | city-mpg | Non | Caractéristique technique, pas de leakage |
    | horsepower | Non | Corrélée au risque, connue a la souscription |

    Note : normalized-losses est problématique car c'est une mesure de risque passée —
    si elle n'est pas disponible au moment de la prédiction (nouveau véhicule), elle
    doit etre exclue ou imputée.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    8.4 — Application du RobustScaler sur un nouveau vehicule (price = 45 000 $) :

    La bonne pratique est d'appeler scaler.transform([[45000]]) sur le scaler déjà
    fitte — sans le re-fitter. Le scaler a mémorisé la mediane et l'IQR du train set
    lors du fit_transform initial.

    Erreur classique à éviter : appeler scaler.fit_transform([[45000]]) sur la nouvelle
    valeur seule. Cela recalculerait la médiane et l'IQR sur un seul point, ce qui est
    sans sens et introduit une inconsistance entre le train et la production —
    c'est du data leakage inverse.

    En pratique : sauvegarder le scaler avec joblib.dump(scaler, 'scaler.pkl') en
    production et le recharger avec joblib.load pour chaque nouvelle prédiction.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    8.5 — Resume du pipeline de nettoyage en 5 etapes :


    | Etape | Description | Outil scikit-learn |
    |---|---|---|
    | 1. Audit | Analyse des types, manquants, doublons, statistiques | pandas (isnull, dtypes, describe) |
    | 2. Imputation | Mediane par groupe (MAR), mode (MCAR), KNNImputer (MAR technique) | SimpleImputer, KNNImputer |
    | 3. Detection outliers | IQR + Z-score + Isolation Forest, decision de conservation | IsolationForest, scipy.stats |
    | 4. Scaling | RobustScaler sur les features numeriques (robuste aux outliers legitimes) | RobustScaler |
    | 5. Pipeline complet | ColumnTransformer + train/test split sans leakage, sauvegarde | ColumnTransformer, Pipeline, joblib |
    """)
    return


if __name__ == "__main__":
    app.run()
