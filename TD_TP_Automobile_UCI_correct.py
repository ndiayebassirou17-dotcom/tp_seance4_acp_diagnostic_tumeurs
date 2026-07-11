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

    return (
        IsolationForest,
        KNNImputer,
        MinMaxScaler,
        Pipeline,
        RobustScaler,
        SimpleImputer,
        StandardScaler,
        mo,
        np,
        pd,
        plt,
        stats,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md("""
    # TD/TP — Dataset Automobile (UCI ML Repository)
    ## Cours : Collecte, Nettoyage, Preparation et Echantillonnage

    Contexte metier : vous etes data analyst dans une entreprise d'assurance automobile.
    Mission : nettoyer, preparer et analyser les caracteristiques de 205 vehicules (1985)
    pour modeliser le risque d'assurance (symboling).
    """)
    return


@app.cell
def _():
    import urllib.request
    import zipfile
    import os
    import glob

    DATA_DIR = "datasets/raw"
    DATA_FILE = "/home/bachir/Documents/data_analyste/datasets/raw/automobile/imports-85.data"
    DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)
    ZIP_URL = "https://archive.ics.uci.edu/static/public/10/automobile.zip"
    ZIP_PATH = os.path.join(DATA_DIR, "automobile.zip")

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.isfile(DATA_PATH):
        print("Telechargement en cours...")
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(DATA_DIR)
        if os.path.isfile(ZIP_PATH):
            os.remove(ZIP_PATH)

    found = os.path.isfile(DATA_PATH)
    if found:
        size = os.path.getsize(DATA_PATH)
        print(f"Fichier pret : {DATA_PATH} ({size} octets)")
    else:
        data_candidates = glob.glob(os.path.join(DATA_DIR, "*.data"))
        print(f"imports-85.data non trouve. Fichiers .data trouves : {data_candidates}")
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

    assert len(df) == 205, f"ERREUR : {len(df)} lignes chargees au lieu de 205 — mauvais fichier ?"
    assert len(df.columns) == 26, f"ERREUR : {len(df.columns)} colonnes au lieu de 26"
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
    1.2 — Incoherences de types observees :

    Les colonnes 'horsepower', 'peak-rpm', 'bore', 'stroke' et 'price' devraient etre
    numeriques (float), mais elles sont chargees en 'object' si les valeurs '?' ne sont
    pas remplacees par NaN avant le chargement. Avec na_values=['?'], pandas les convertit
    correctement en float. La colonne 'normalized-losses' est aussi en float alors qu'elle
    pourrait sembler entiere. La colonne 'num-of-cylinders' est une chaine de caracteres
    ('four', 'six', etc.) alors qu'elle represente un entier — c'est une incoherence
    notable a corriger pour la modelisation.
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
    1.3 — Les 3 colonnes les plus touchees par les valeurs manquantes sont
    'normalized-losses' (~20%), 'bore' et 'stroke' (~2%), puis 'horsepower', 'peak-rpm'
    et 'price' (~1-2%). 'normalized-losses' est de loin la plus problematique car elle
    represente la variable cible actuarielle — son imputation doit etre rigoureuse.
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
    1.5 — Observation sur 'compression-ratio' : la colonne presente un min autour de 7
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
    certaines marques n'ont aucune valeur observee, d'autres en ont toutes. Ce pattern
    non homogene indique que le manquant depend d'une variable observee ('make') et non
    du hasard pur — c'est un mecanisme MAR (Missing At Random). Le calcul actuariel des
    pertes normalisees est realise marque par marque, donc les marques avec peu de
    sinistres n'ont pas de valeur calculable. Strategie recommandee : imputation par
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
    types de carrosserie specifiques. Ce pattern suggere un mecanisme MAR : le prix
    manque davantage pour certaines marques (ex: marques peu courantes dont le catalogue
    n'etait pas disponible en 1985). Le manquant depend d'une variable observee ('make'),
    pas du prix lui-meme. Strategie : imputation par la mediane par marque, avec creation
    d'une indicatrice 'price_imputed'.
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
    'stroke', 'horsepower' et 'peak-rpm' tendent a manquer ensemble. Une correlation
    elevee entre deux colonnes signifie que quand l'une manque, l'autre aussi —
    generalement lie au fait que ces donnees techniques proviennent de la meme source
    (fiche technique constructeur). Si la correlation est forte, un KNNImputer utilisant
    les colonnes voisines sera plus efficace qu'une imputation par mediane simple.

    Tableau recapitulatif des mecanismes :

    | Colonne | Mecanisme | Justification | Strategie |
    |---|---|---|---|
    | normalized-losses | MAR | Depend de 'make' (calcul par marque) | Mediane par marque |
    | num-of-doors | MCAR | 2 valeurs, aucune logique apparente | Mode global |
    | bore | MAR | Donnee technique, depend du moteur | KNNImputer ou mediane |
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(nl_before, bins=20, color="steelblue")
    axes[0].set_title("normalized-losses avant imputation")
    axes[1].hist(nl_after, bins=20, color="darkorange")
    axes[1].set_title("normalized-losses apres imputation")
    plt.tight_layout()
    plt.savefig("datasets/raw/nl_imputation.png", dpi=80)
    plt.close()
    fig
    return (df_imputed,)


@app.cell
def _(mo):
    mo.md("""
    3.1 — L'imputation de 'normalized-losses' par la mediane par marque preserve la
    structure du dataset : chaque marque garde une valeur coherente avec son propre
    historique de sinistres. Pour les marques sans aucune valeur observee, la mediane
    globale est utilisee en repli. L'histogramme apres imputation montre une distribution
    plus complete mais dont la forme generale est preservee — c'est le signe d'une
    imputation de qualite.
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
def _(KNNImputer, df_imputed, mo):
    cols_hp = ["horsepower", "peak-rpm", "engine-size"]

    imp_knn = KNNImputer(n_neighbors=5)
    hp_knn = imp_knn.fit_transform(df_imputed[cols_hp])

    n_hp = df_imputed["horsepower"].isnull().sum()
    n_rpm = df_imputed["peak-rpm"].isnull().sum()

    df_imputed["horsepower"] = hp_knn[:, 0]
    df_imputed["peak-rpm"] = hp_knn[:, 1]

    mo.md(f"""
    3.3 — Comparaison SimpleImputer vs KNNImputer pour 'horsepower' ({n_hp} valeurs) et 'peak-rpm' ({n_rpm} valeurs) :

    La mediane globale (SimpleImputer) ignore la relation entre la puissance et la cylindree.
    Le KNNImputer, en utilisant les 5 voisins les plus proches sur ['horsepower', 'peak-rpm',
    'engine-size'], tient compte de ces correlations et produit des estimations plus
    realistes — un moteur de grande cylindree aura une puissance imputee plus elevee.
    On retient le KNNImputer pour la coherence physique.
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
    utile pour la modelisation car elle capture le fait que le prix etait inconnu — ce
    qui peut lui-meme etre un signal predictif (les vehicules sans prix catalogue sont
    souvent des modeles rares ou importes, potentiellement plus risques ou plus chers).
    En l'incluant comme feature, le modele peut apprendre cet effet sans confondre une
    valeur imputee avec une valeur reelle.
    """)
    return


@app.cell
def _(df_imputed, mo):
    df_imputed["bore"] = df_imputed["bore"].fillna(df_imputed["bore"].median())
    df_imputed["stroke"] = df_imputed["stroke"].fillna(df_imputed["stroke"].median())

    total_final = df_imputed.isnull().sum().sum()
    mo.md(f"3.5 — Verification finale : df_imputed.isnull().sum().sum() = {total_final}. Toutes les valeurs manquantes ont ete imputees.")
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
            "n_outliers": mask.sum()
        })

    cols_out = ["price", "horsepower", "normalized-losses"]
    iqr_results = pd.DataFrame({c: iqr_outliers(df_imputed[c]) for c in cols_out}).T
    iqr_results
    return (cols_out,)


@app.cell
def _(cols_out, df_imputed, mo, np, pd, stats):
    def zscore_outliers(series):
        z = np.abs(stats.zscore(series.dropna()))
        return (z > 3).sum()

    z_results = pd.Series({c: zscore_outliers(df_imputed[c]) for c in cols_out}, name="n_outliers_zscore")
    mo.md(f"""
    4.2 — Comparaison IQR vs Z-score :

    {z_results.to_string()}

    L'IQR detecte généralement plus d'outliers que le Z-score sur des distributions
    asymétriques comme 'price'. Le Z-score est sensible a la moyenne et l'écart-type
    qui sont eux-memes influences par les valeurs extremes, ce qui dilate l'échelle et
    reduit artificiellement le nombre de points détectes. L'IQR, base sur les
    quantiles et  est robuste à cette asymetrie.
    """)
    return


@app.cell
def _(IsolationForest, df_imputed, mo):
    cols_iso = ["price", "horsepower", "engine-size", "curb-weight", "normalized-losses"]
    iso = IsolationForest(contamination=0.05, random_state=42)
    df_imputed["iso_label"] = iso.fit_predict(df_imputed[cols_iso])
    n_iso_outliers = (df_imputed["iso_label"] == -1).sum()
    mo.md(f"4.3 — Isolation Forest (contamination=0.05) detecte {n_iso_outliers} outliers multivaries sur les colonnes numeriques selectionnees.")
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
    un prix élève réflete un positionnement premium, pas une erreur de saisie.
    Stratégie adoptee : conservation. Supprimer ces vehicules biaiserait le modèle
    en excluant les voitures haut de gamme qui ont probablement un profil de risque
    distinctif. On peut envisager une transformation log(price) pour reduire l'asymétrie.

    Marques concernees :
    {marques_outliers.to_string()}
    """)
    return IQR_p, Q3_p


@app.cell
def _(IQR_p, Q3_p, df_imputed, plt):
    df_no_out = df_imputed[df_imputed["price"] <= Q3_p + 1.5 * IQR_p]

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
    plt.savefig("datasets/raw/price_outliers.png", dpi=80)
    plt.close()
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
    5.1 — Les plages de valeurs sont tres heterogenes : 'price' va de ~5000 a ~45000,
    'height' de ~47 a ~60 (pouces), 'horsepower' de ~50 a ~300. Sans mise a l'echelle,
    les algorithmes bases sur des distances (KNN, SVM, regression reguliee) seraient
    domines par les colonnes a grande variance, biaisant les resultats. La normalisation
    est donc indispensable avant toute modelisation.
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
    plt.savefig("datasets/raw/scalers_price.png", dpi=80)
    plt.close()
    fig3
    return


@app.cell
def _(mo):
    mo.md("""
    5.2 — Les trois scalers produisent des distributions différentes :
    MinMaxScaler compresse tout dans [0,1] mais reste sensible aux outliers (un prix
    extremement élevé tire tout vers 0). StandardScaler centre et reduit mais les outliers
    restent presents et eloignes. RobustScaler utilise la mediane et l'IQR — il est
    insensible aux valeurs extremes et produit une distribution plus equilibrée.

    5.3 — Pour ce dataset avec des outliers légitimes (voitures de luxe), le RobustScaler
    est recommandé. Il ne pénalise pas les véhicules premium en les traitant comme des
    erreurs, tout en permettant un scaling cohérent pour les algorithmes de ML.
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
    5.4 — Après application du RobustScaler, la mediane de chaque colonne est
    approximativement 0 (par definition du scaler). C'est la vérification attendue :
    une valeur exactement a la mediane originale est transformée en 0.
    """)
    return


@app.cell
def _(df_imputed, df_scaled, mo):
    corr_avant = df_imputed["city-mpg"].corr(df_imputed["horsepower"])
    corr_apres = df_scaled["city-mpg"].corr(df_scaled["horsepower"])
    mo.md(f"""
    5.5 — Correlation de Pearson entre 'city-mpg' et 'horsepower' :
    - Avant scaling : {corr_avant:.4f}
    - Apres RobustScaler : {corr_apres:.4f}

    La correlation lineaire est parfaitement preservee apres un scaling monotone
    (MinMax, Standard, Robust). En effet, ces transformations sont des fonctions
    affines qui ne modifient pas les relations lineaires entre variables. La
    correlation negative confirme la logique physique : plus un vehicule est puissant,
    moins il est econome en carburant.
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
def _(Pipeline, RobustScaler, SimpleImputer, cat_features, num_features):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    from sklearn.compose import ColumnTransformer as CT
    preprocessor = CT(transformers=[
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

    Distribution de symboling dans la population :
    {distrib_pop.round(3).to_string()}

    Distribution de symboling dans l'echantillon SRS :
    {distrib_srs.round(3).to_string()}

    Couverture fuel-type population :
    {couverture_fuel_pop.round(1).to_string()}

    Couverture fuel-type echantillon SRS :
    {couverture_fuel_srs.round(1).to_string()}

    L'echantillon peut ne pas etre representatif car avec n=50 sur 205 lignes,
    les categories minoritaires (ex: diesel ~10% = environ 20 vehicules) peuvent
    etre sous- ou sur-echantillonnees par simple variabilite aleatoire. Le SRS ne
    garantit pas la proportionnalite des strates.
    """)
    return couverture_fuel_pop, couverture_fuel_srs


@app.cell
def _(couverture_fuel_pop, couverture_fuel_srs, df, mo, train_test_split):
    df_clean_fuel = df.dropna(subset=["fuel-type"])
    _, strat_sample = train_test_split(
        df_clean_fuel,
        test_size=50/len(df_clean_fuel),
        stratify=df_clean_fuel["fuel-type"],
        random_state=42
    )

    couverture_fuel_strat = strat_sample["fuel-type"].value_counts(normalize=True) * 100

    mo.md(f"""
    7.2 — Echantillonnage stratifie sur fuel-type (n~50) :

    Repartition fuel-type (population) :
    {couverture_fuel_pop.round(1).to_string()}

    Repartition fuel-type (SRS) :
    {couverture_fuel_srs.round(1).to_string()}

    Repartition fuel-type (stratifie) :
    {couverture_fuel_strat.round(1).to_string()}

    L'echantillonnage stratifie garantit que chaque strate (fuel-type) est
    representee proportionnellement. Il est preferable au SRS quand on s'interesse
    a des sous-groupes minoritaires (ex: diesel) ou quand la variable d'interet
    est correlee avec la variable de stratification.
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
    7.3 — Echantillonnage par grappes (marques comme grappes) :

    Nombre de marques distinctes : {n_marques}
    Marques tirees : {list(marques_choisies)}
    Vehicules obtenus : {n_cluster}

    Couverture body-style (population) :
    {couverture_body_pop.round(1).to_string()}

    Couverture body-style (echantillon grappes) :
    {couverture_body.round(1).to_string()}

    Inconvenient majeur : les vehicules au sein d'une meme marque se ressemblent
    (homogeneite intra-grappe) — l'echantillon peut manquer de diversite si les
    marques selectionnees sont atypiques. De plus, la taille de l'echantillon
    n'est pas controlee.
    """)
    return


@app.cell
def _(df, mo):
    df_rich = df[df["price"] > 20000]
    pct_rich = len(df_rich) / len(df.dropna(subset=["price"])) * 100
    sym_risque_pop = (df["symboling"] >= 2).mean() * 100
    sym_risque_rich = (df_rich["symboling"] >= 2).mean() * 100

    mo.md(f"""
    7.4 — Biais d'echantillonnage (price > 20 000 $) :

    Pourcentage du dataset concerne : {pct_rich:.1f}%
    Taux de symboling >= 2 dans la population : {sym_risque_pop:.1f}%
    Taux de symboling >= 2 dans l'echantillon biaise : {sym_risque_rich:.1f}%

    Ce type de biais s'appelle biais de selection : l'enqueteur ne capture qu'une
    sous-population non representative (les vehicules chers). Les vehicules abordables
    sont totalement exclus, ce qui fausse toute conclusion sur la distribution du
    risque d'assurance dans la population generale.
    """)
    return


@app.cell
def _(df, mo):
    distrib_sym = df["symboling"].value_counts().sort_index()
    classe_maj = distrib_sym.idxmax()
    pct_maj = distrib_sym.max() / len(df) * 100

    mo.md(f"""
    7.5 — Classes desequilibrees pour symboling :

    {distrib_sym.to_string()}

    Classe majoritaire : {classe_maj} ({pct_maj:.1f}% du dataset)

    Deux strategies pour gerer ce desequilibre sans SMOTE :
    1. Sous-echantillonnage (undersampling) : reduire aleatoirement les classes
       majoritaires pour les ramener au niveau de la classe minoritaire. Simple
       mais entraine une perte d'information.
    2. Ponderation des classes (class_weight='balanced') : attribuer un poids
       inversement proportionnel a la frequence de chaque classe dans l'algorithme
       de classification. Le modele penalise davantage les erreurs sur les classes
       rares sans supprimer de donnees.
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

    mo.md(f"""
    8.1 — Conversion de 'num-of-cylinders' (textuel -> numerique) :

    Approche 1 — Mapping manuel : utiliser un dictionnaire
    cylinders_map = {{'two': 2, 'three': 3, 'four': 4, ...}} puis df['num-of-cylinders'].map(cylinders_map).
    Simple, explicite, sans ambiguite. Recommandee quand le vocabulaire est fige.

    Approche 2 — OrdinalEncoder de scikit-learn avec les categories ordonnees explicitement.
    Integrable directement dans un Pipeline.

    Valeurs uniques presentes : {sorted(df['num-of-cylinders'].dropna().unique())}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    8.2 — Normalisation de 'compression-ratio' avec distribution bimodale :

    StandardScaler est inapproprie car il suppose une distribution unimodale
    symetrique — la moyenne serait entre les deux pics (autour de 14) alors qu'aucun
    vehicule n'a ce ratio, ce qui rendrait le scaling sans sens physique.

    RobustScaler est plus robuste mais souffre du meme probleme fondamental.

    Meilleure approche : creer une variable binaire 'is_diesel' (1 si fuel-type=diesel,
    0 sinon) et normaliser separement les deux sous-populations. Alternativement,
    utiliser 'compression-ratio' comme feature categorielle discretisee (bas/haut)
    ou l'inclure avec 'fuel-type' dans une interaction.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    8.3 — Colonnes a exclure pour predire symboling :

    | Colonne | Exclure ? | Raison |
    |---|---|---|
    | normalized-losses | Avec prudence | Correlee a symboling, risque de leakage si non disponible a la prediction |
    | price | Non | Feature legitime, connu a la souscription |
    | make | Non (avec encodage) | Connu a la souscription, informatif sur le profil de risque |
    | city-mpg | Non | Caracteristique technique connue, pas de leakage |
    | horsepower | Non | Caracteristique technique connue, correlee au risque |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    8.4 — Application du RobustScaler sur un nouveau vehicule (price = 45 000 $) :

    La bonne pratique est d'appeler scaler.transform([[45000]]) sur le scaler deja
    fitte — sans le re-fitter. Le scaler a memorise la mediane et l'IQR du train set
    lors du fit_transform initial.

    Erreur classique a eviter : appeler scaler.fit_transform([[45000]]) sur la nouvelle
    valeur seule. Cela recalculerait la mediane et l'IQR sur un seul point, ce qui est
    sans sens et introduit une inconsistance entre le train et la production —
    c'est du data leakage inverse.

    En pratique : sauvegarder le scaler avec joblib.dump(scaler, 'scaler.pkl')
    et le recharger avec joblib.load pour chaque nouvelle prediction.
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
