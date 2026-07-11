import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from scipy import stats

    return mo, np, pd, stats


@app.cell
def _(mo):
    mo.md("""
    # Séance 2 — Collecte, nettoyage, préparation des données
    ## TD/TP — Dataset Titanic (titanic3, 1309 passagers, 14 colonnes)

    Vous êtes data analyst pour une compagnie maritime. Mission : nettoyer et préparer
    les données pour analyser les chances de survie du naufrage du RMS Titanic.
    """)
    return


@app.cell
def _(pd):
    df_titanic = pd.read_csv("datasets/raw/titanic.csv")
    df_titanic
    return (df_titanic,)


@app.cell
def _(mo):
    mo.md("""
    ## Exercice 1 — Audit du dataset
    """)
    return


@app.cell
def _(df_titanic, mo):
    n_rows, n_cols = df_titanic.shape
    mo.md(f"Lignes : {n_rows} — Colonnes : {n_cols}")
    return


@app.cell
def _(df_titanic):
    missing_pct = (df_titanic.isnull().mean() * 100).sort_values(ascending=False)
    missing_pct
    return


@app.cell
def _(mo):
    mo.md("""
    Réponse 'body' présente le taux de manquants le plus élevé (~90%).
    C'est logique : cette colonne n'a de sens que pour les passagers décédés
    dont le corps a été repêché et identifié. La grande majorité des passagers
    (survivants + corps jamais retrouvés) n'ont donc structurellement aucune valeur
    possible ici — ce n'est pas un défaut de collecte, c'est la nature de la variable.
    """)
    return


@app.cell
def _(df_titanic):
    df_titanic.dtypes
    return


@app.cell
def _(mo):
    mo.md("""
    Incohérences de types observées :
    - 'body' est stocké en 'float64' alors que c'est un identifiant entier
      (numéro de corps). Pandas force le passage en float dès qu'une colonne
      numérique contient des 'NaN'.
    - 'pclass', 'sibsp', 'parch', 'survived' sont des entiers mais ont en réalité
      un statut catégoriel/ordinal ('pclass') ou binaire ('survived').
    - 'cabin' et 'ticket' sont des chaînes alphanumériques qui mélangent lettres
      et chiffres (ex: 'C85', 'PC 17599') — à parser si on veut en extraire de
      l'information (lettre de pont, préfixe de billet).
    """)
    return


@app.cell
def _(df_titanic):
    n_duplicates = df_titanic.duplicated().sum()
    n_duplicates_name_only = df_titanic.duplicated(subset=["name"]).sum()
    n_duplicates, n_duplicates_name_only
    return n_duplicates, n_duplicates_name_only


@app.cell
def _(mo, n_duplicates, n_duplicates_name_only):
    mo.md(f"""
    Réponse : {n_duplicates} doublon(s) strict(s) (toutes colonnes identiques).
    On compte {n_duplicates_name_only} doublon(s) sur le 'name' seul — ce ne sont
    pas forcément des erreurs : il peut s'agir d'homonymes, donc avant de les
    supprimer il faut vérifier les autres colonnes (âge, billet, famille) pour
    confirmer qu'il s'agit bien de la même personne.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Exercice 2 — Mécanismes de données manquantes

    | Colonne    | % Manquants | Mécanisme | Justification | Stratégie |
    |---|---|---|---|---|
    | 'age'      | ~20% | MAR  | Dépend de variables observées ('pclass', 'sex') — les passagers de 3ᵉ classe sont moins souvent documentés | Imputation par médiane groupée (pclass × sex) |
    | 'cabin'    | ~77% | MAR (proche MNAR) | Dépend fortement de 'pclass' (3ᵉ classe = pas de cabine attribuée/enregistrée) | Ne pas imputer une valeur — créer 'has_cabin' |
    | 'embarked' | <1%  | MCAR | 2 valeurs manquantes, aucune logique apparente, simple oubli administratif | Imputation par le mode |
    | 'fare'     | <1%  | MCAR | 1 seule valeur manquante, cas isolé | Imputation par médiane (idéalement par pclass) |
    | 'boat'     | ~63% | MNAR | Le manque dépend directement de la variable cible : pas de canot ≈ n'a pas survécu | Ne pas imputer — créer 'has_boat' |
    | 'body'     | ~90% | MNAR | Dépend de 'survived' ET du fait que le corps ait été repêché (deux facteurs non observés) | Ne pas imputer — créer 'body_recovered' |

    Point clé pour 'boat' et 'body' : ce ne sont pas de "vraies" données manquantes
    au hasard — leur absence encode de l'information sur 'survived'. Les utiliser
    telles quelles comme features dans un modèle de survie créerait une fuite de
    données (data leakage) massive.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Exercice 3 — Comparer deux stratégies de nettoyage
    """)
    return


@app.cell
def _(df_titanic):
    df_a = df_titanic.dropna(subset=["age", "embarked", "fare"])

    n_a = len(df_a)
    pct_loss_a = 100 * (1 - n_a / len(df_titanic))
    survival_a = df_a["survived"].mean()
    sex_a = df_a["sex"].value_counts(normalize=True) * 100
    return n_a, pct_loss_a, sex_a, survival_a


@app.cell
def _(mo, n_a, pct_loss_a, sex_a, survival_a):
    mo.md(f"""
    Stratégie A (suppression) :
    - Passagers conservés : {n_a} ({pct_loss_a:.1f}% perdus)
    - Taux de survie global : {survival_a:.3f}
    - Répartition H/F conservée : {sex_a.round(1).to_dict()}
    """)
    return


@app.cell
def _(df_titanic):
    df_b = df_titanic.copy()

    df_b["age"] = df_b.groupby(["pclass", "sex"])["age"].transform(
        lambda x: x.fillna(x.median())
    )
    df_b["embarked"] = df_b["embarked"].fillna(df_b["embarked"].mode()[0])
    df_b["fare"] = df_b["fare"].fillna(df_b["fare"].median())
    df_b["has_cabin"] = df_b["cabin"].notna().astype(int)

    n_b = len(df_b)
    survival_b = df_b["survived"].mean()
    sex_b = df_b["sex"].value_counts(normalize=True) * 100
    return n_b, sex_b, survival_b


@app.cell
def _(mo, n_b, sex_b, survival_b):
    mo.md(f"""
    Stratégie B (imputation) :
    - Passagers conservés : {n_b} (0% perdus)
    - Taux de survie global : {survival_b:.3f}
    - Répartition H/F conservée : {sex_b.round(1).to_dict()}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    Comparaison et biais :

    La stratégie A perd environ 20% des lignes (essentiellement à cause de 'age').
    Oui, A introduit un biais. Les valeurs manquantes de 'age' ne sont pas
    aléatoires (MAR, liées à 'pclass') : elles sont plus fréquentes en 3ᵉ classe,
    qui a aussi un taux de survie plus faible. En supprimant ces lignes, on
    sur-représente mécaniquement les 1ʳᵉ et 2ᵉ classes, donc le taux de survie
    calculé sur A est artificiellement plus élevé que la réalité globale — un
    biais de sélection classique. La stratégie B, en imputant intelligemment,
    conserve toute la population et donne une estimation plus fidèle.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Exercice 4 — Détection et traitement des outliers ('fare')
    """)
    return


@app.cell
def _(df_titanic):
    Q1 = df_titanic["fare"].quantile(0.25)
    Q3 = df_titanic["fare"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_iqr = df_titanic[
        (df_titanic["fare"] < lower_bound) | (df_titanic["fare"] > upper_bound)
    ]
    n_outliers_iqr = len(outliers_iqr)
    return (n_outliers_iqr,)


@app.cell
def _(df_titanic, np, stats):
    fare_clean = df_titanic["fare"].dropna()
    z_scores = np.abs(stats.zscore(fare_clean))
    outliers_z = fare_clean[z_scores > 3]
    n_outliers_z = len(outliers_z)
    return (n_outliers_z,)


@app.cell
def _(mo, n_outliers_iqr, n_outliers_z):
    mo.md(f"""
    - Outliers détectés par IQR : {n_outliers_iqr}
    - Outliers détectés par Z-score (|z| > 3) : {n_outliers_z}

    Différence et interprétation :

    L'IQR détecte généralement beaucoup plus d'outliers que le Z-score, car
    'fare' est très asymétrique (longue queue à droite). La méthode IQR est robuste
    à l'asymétrie, contrairement au Z-score qui repose sur la moyenne et l'écart-type,
    eux-mêmes "tirés" vers le haut par les valeurs extrêmes — ce qui réduit
    artificiellement le nombre de points repérés comme outliers.

    Erreurs ou valeurs légitimes ? Ce sont des valeurs légitimes. Les tarifs
    très élevés correspondent à des suites de 1ʳᵉ classe (familles ayant payé pour
    plusieurs cabines regroupées sous un même tarif). Les tarifs à 0 correspondent
    souvent à des membres d'équipage ou du personnel avec billet gratuit.

    Stratégie recommandée : conservation, éventuellement avec une transformation
    log ('np.log1p(fare)') pour réduire l'asymétrie avant modélisation, plutôt que
    suppression ou remplacement, qui ferait perdre une information réelle et
    prédictive (le tarif est corrélé à la classe et donc à la survie).
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Exercice 5 — Discussion
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    1. 'body' : MCAR, MAR ou MNAR ?

    MNAR. 'body' n'est observé que pour les passagers décédés et dont le corps
    a été repêché et identifié. Son absence dépend donc directement de 'survived'
    (la variable cible) et d'un second facteur non observé (le corps a-t-il été
    retrouvé). On ne peut pas "tester" ce mécanisme avec les données observées —
    c'est la définition même de MNAR.

    2. Biais introduit en supprimant les passagers sans 'cabin' (vs 'pclass') ?

    'cabin' est renseigné presque uniquement en 1ʳᵉ classe (et un peu en 2ᵉ).
    Supprimer les lignes sans cabine reviendrait à éliminer presque toute la
    3ᵉ classe, faussant gravement toute analyse liant classe sociale et survie —
    un biais de sélection encore plus fort que pour 'age'.

    3. Faut-il garder 'boat', 'body', 'ticket', 'name' pour un modèle prédictif ?

    - 'boat' et 'body': non, à exclure absolument — fuite de données, car leur
      valeur n'est connue qu'après le naufrage et encode quasi-directement 'survived'.
    - 'ticket' et 'name' : la colonne brute a peu de valeur prédictive directe, mais
      on peut en extraire des features utiles : le titre dans 'name'
      (Mr/Mrs/Miss/Master, corrélé à l'âge et au statut marital) et la taille du
      groupe partageant le même 'ticket' (proxy de voyage en famille). Garder la
      colonne brute = inutile ; garder les features dérivées = pertinent.

    4. Médiane globale vs médiane par pclass+sex pour 'age' ?

    La médiane globale écrase les différences réelles entre groupes : la 1ʳᵉ classe
    comptait des passagers globalement plus âgés et plus de femmes au foyer, la
    3ᵉ classe plus de jeunes hommes immigrants. Imputer par groupe préserve la
    structure et la variance réelle des données, alors que l'imputation globale
    comprime artificiellement la distribution vers une seule valeur centrale,
    atténuant les écarts entre classes — ce qui peut affaiblir le pouvoir prédictif
    d''age' dans un modèle final.
    """)
    return


if __name__ == "__main__":
    app.run()
