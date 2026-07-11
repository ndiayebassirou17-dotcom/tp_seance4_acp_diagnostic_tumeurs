import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import hashlib
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline

    return (
        LogisticRegression,
        PCA,
        Pipeline,
        StandardScaler,
        cross_val_score,
        hashlib,
        load_breast_cancer,
        mo,
        np,
        plt,
        sns,
    )


@app.cell
def _(mo):
    mo.md("""
    Corrigé — TP/TD : L'ACP appliquée au diagnostic de tumeurs

    Notebook de correction complète : réponses des QCM (Partie A) et code
    des exercices pratiques (Partie B).
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    🔧 Fonction de vérification des QCM
    """)
    return


@app.cell
def _(hashlib, mo):
    """Fonction de vérification des QCM — réponses comparées par hachage (pas de spoiler)."""

    def verifier_qcm(valeur, hash_attendu):
        if valeur is None:
            return mo.callout(
                mo.md("⏳ *Sélectionnez une réponse ci-dessus pour lancer la vérification.*"),
                kind="neutral",
            )
        _h = hashlib.sha256(valeur.strip().upper().encode()).hexdigest()[:8]
        if _h == hash_attendu:
            return mo.callout(mo.md("✅ Bonne réponse !"), kind="success")
        return mo.callout(
            mo.md("❌ Réponse incorrecte. Revoyez la Partie 2 du cours."),
            kind="danger",
        )

    return (verifier_qcm,)


@app.cell
def _(mo):
    mo.md("""
    📚 Partie A — Rappel de cours (Exercices 1 → 12)
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ✏️ Exercice 1 — Qu'est-ce qu'une composante principale ?
    Réponse : B

    Une composante principale est une combinaison linéaire des variables,
    non corrélée aux autres composantes, ordonnée par variance décroissante.
    """)
    return


@app.cell
def _(mo):
    ex1 = mo.ui.radio(options=["A", "B", "C", "D"], value="B", label="Votre réponse :")
    ex1
    return (ex1,)


@app.cell
def _(ex1, verifier_qcm):
    verifier_qcm(ex1.value, "df7e70e5")
    return


@app.cell
def _(mo):
    mo.md("""
    ✏️ Exercice 2 — ACP : supervisée ou non ?
    Réponse : B — L'ACP est non supervisée : elle ne regarde jamais les étiquettes.
    """)
    return


@app.cell
def _(mo):
    ex2 = mo.ui.radio(options=["A", "B", "C", "D"], value="B", label="Votre réponse :")
    ex2
    return (ex2,)


@app.cell
def _(ex2, verifier_qcm):
    verifier_qcm(ex2.value, "df7e70e5")
    return


@app.cell
def _(mo):
    mo.md("""
    ✏️ Exercice 3 — Pourquoi standardiser avant une ACP ?
    Réponse : A — L'ACP maximise la variance : sans standardisation,
    une variable à grande échelle domine mécaniquement PC1.
    """)
    return


@app.cell
def _(mo):
    ex3 = mo.ui.radio(options=["A", "B", "C", "D"], value="A", label="Votre réponse :")
    ex3
    return (ex3,)


@app.cell
def _(ex3, verifier_qcm):
    verifier_qcm(ex3.value, "559aead0")
    return


@app.cell
def _(mo):
    mo.md("""
    ✏️ Exercice 4 — La matrice de covariance des données standardisées
    Réponse : D — Sur données standardisées, Σ est la matrice de
    corrélation (diagonale = 1).
    """)
    return


@app.cell
def _(mo):
    ex4 = mo.ui.radio(options=["A", "B", "C", "D"], value="D", label="Votre réponse :")
    ex4
    return (ex4,)


@app.cell
def _(ex4, verifier_qcm):
    verifier_qcm(ex4.value, "3f39d5c3")
    return


@app.cell
def _(mo):
    mo.md("""
    ✏️ Exercice 5 — Vecteurs propres vs valeurs propres
    Réponse : B — direction = vecteur propre de Σ ; importance = valeur propre λₖ.
    """)
    return


@app.cell
def _(mo):
    ex5 = mo.ui.radio(options=["A", "B", "C", "D"], value="B", label="Votre réponse :")
    ex5
    return (ex5,)


@app.cell
def _(ex5, verifier_qcm):
    verifier_qcm(ex5.value, "df7e70e5")
    return


@app.cell
def _(mo):
    mo.md("""
    ✏️ Exercice 6 — La direction de variance maximale
    Réponse : A — le vecteur propre associé à la plus grande valeur propre.
    """)
    return


@app.cell
def _(mo):
    ex6 = mo.ui.radio(options=["A", "B", "C", "D"], value="A", label="Votre réponse :")
    ex6
    return (ex6,)


@app.cell
def _(ex6, verifier_qcm):
    verifier_qcm(ex6.value, "559aead0")
    return


@app.cell
def _(mo):
    mo.md("""
    ✏️ Exercice 7 — La dualité de Pearson (1901)
    Réponse : C — maximiser la variance projetée revient à minimiser
    la somme des distances quadratiques points–projections.
    """)
    return


@app.cell
def _(mo):
    ex7 = mo.ui.radio(options=["A", "B", "C", "D"], value="C", label="Votre réponse :")
    ex7
    return (ex7,)


@app.cell
def _(ex7, verifier_qcm):
    verifier_qcm(ex7.value, "6b23c0d5")
    return


@app.cell
def _(mo):
    mo.md("""
    ✏️ Exercice 8 — Trace et déterminant
    Réponse : A — trace(Σ) = Σλₖ (somme des variances) et det(Σ) = Πλₖ.
    """)
    return


@app.cell
def _(mo):
    ex8 = mo.ui.radio(options=["A", "B", "C", "D"], value="A", label="Votre réponse :")
    ex8
    return (ex8,)


@app.cell
def _(ex8, verifier_qcm):
    verifier_qcm(ex8.value, "559aead0")
    return


@app.cell
def _(mo):
    mo.md("""
    ✏️ Exercice 9 — Le critère de Kaiser
    Réponse : B — on conserve les composantes telles que λₖ > 1.
    """)
    return


@app.cell
def _(mo):
    ex9 = mo.ui.radio(options=["A", "B", "C", "D"], value="B", label="Votre réponse :")
    ex9
    return (ex9,)


@app.cell
def _(ex9, verifier_qcm):
    verifier_qcm(ex9.value, "df7e70e5")
    return


@app.cell
def _(mo):
    mo.md("""
    ✏️ Exercice 10 — Scores vs loadings
    Réponse : C — Z = coordonnées des individus dans le nouvel espace ;
    Wₖ = contribution des variables d'origine aux composantes.
    """)
    return


@app.cell
def _(mo):
    ex10 = mo.ui.radio(options=["A", "B", "C", "D"], value="C", label="Votre réponse :")
    ex10
    return (ex10,)


@app.cell
def _(ex10, verifier_qcm):
    verifier_qcm(ex10.value, "6b23c0d5")
    return


@app.cell
def _(mo):
    mo.md("""
    ✏️ Exercice 11 — Reconstruction et variance perdue
    Réponse : D — X̂ = Z·Wₖᵀ ; 1−PVE(k) = fraction de variance perdue
    (non récupérable).
    """)
    return


@app.cell
def _(mo):
    ex11 = mo.ui.radio(options=["A", "B", "C", "D"], value="D", label="Votre réponse :")
    ex11
    return (ex11,)


@app.cell
def _(ex11, verifier_qcm):
    verifier_qcm(ex11.value, "3f39d5c3")
    return


@app.cell
def _(mo):
    mo.md("""
    ✏️ Exercice 12 — Pourquoi la SVD ?
    Réponse : B — diagonaliser Σ = Xcᵀ Xc est numériquement instable ;
    la SVD agit directement sur Xc, avec λₖ = σₖ²/(n−1).
    """)
    return


@app.cell
def _(mo):
    ex12 = mo.ui.radio(options=["A", "B", "C", "D"], value="B", label="Votre réponse :")
    ex12
    return (ex12,)


@app.cell
def _(ex12, verifier_qcm):
    verifier_qcm(ex12.value, "df7e70e5")
    return


@app.cell
def _(mo):
    mo.md("""
    🧪 Partie B — Pratique (Exercices 13 → 16)
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    📂 Données (fournies)
    """)
    return


@app.cell
def _(StandardScaler, load_breast_cancer):
    """Chargement du dataset Breast Cancer Wisconsin (Diagnostic) — fournie, ne pas modifier."""
    _data = load_breast_cancer()
    X = _data.data
    y = _data.target  # 0 = maligne (cancéreuse), 1 = bénigne
    feat_names = list(_data.feature_names)
    X_std = StandardScaler().fit_transform(X)

    print(f"Breast Cancer Wisconsin (Diagnostic) : {X.shape[0]} biopsies × {X.shape[1]} variables")
    print(f"  Maligne (0) : {(y == 0).sum():>3}  ({(y == 0).mean():.0%})")
    print(f"  Bénigne (1) : {(y == 1).sum():>3}  ({(y == 1).mean():.0%})")
    print(f"\n  Variables (forme des noyaux) : {feat_names[:4]} …")
    return X, X_std, feat_names, y


@app.cell
def _(mo):
    mo.md("""
    🔎 EDA express — les variables sont-elles corrélées ? (fournie)
    """)
    return


@app.cell
def _(X_std, feat_names, np, plt, sns):
    """Matrice de corrélation — cellule fournie."""
    _corr = np.corrcoef(X_std.T)

    _fig, _ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(_corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.2, ax=_ax,
                xticklabels=feat_names, yticklabels=feat_names)
    _ax.set_title("Corrélations entre les 30 variables — Breast Cancer (Diagnostic)")
    _ax.tick_params(axis="x", rotation=90, labelsize=6)
    _ax.tick_params(axis="y", labelsize=6)

    _mask = ~np.eye(len(feat_names), dtype=bool)
    _mean_abs = np.abs(_corr[_mask]).mean()
    plt.tight_layout()
    print(f"Corrélation absolue moyenne (hors diagonale) : {_mean_abs:.2f}")
    print("→ Corrélation élevée : les variables sont très redondantes "
          "(rayon / périmètre / aire notamment) → terrain très favorable à l'ACP.")
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ✏️ Exercice 13 — Combien de composantes faut-il garder ?

    `_var_cum` = `np.cumsum(_var_ratio)`
    """)
    return


@app.cell
def _(PCA, X_std, mo, np, plt):
    _pca_full = PCA().fit(X_std)
    _var_ratio = _pca_full.explained_variance_ratio_

    _var_cum = np.cumsum(_var_ratio)

    _c = range(1, len(_var_ratio) + 1)
    _fig13, _axes = plt.subplots(1, 2, figsize=(13, 4))

    _axes[0].bar(_c, _var_ratio, color="steelblue", edgecolor="white")
    _axes[0].plot(_c, _var_ratio, "ro-", ms=4)
    _axes[0].set_title("Scree plot")
    _axes[0].set_xlabel("Composante")
    _axes[0].set_ylabel("Variance expliquée")
    _axes[0].grid(True, alpha=0.3)

    _axes[1].plot(_c, _var_cum, "go-", lw=2)
    for _t, _col in [(0.80, "orange"), (0.90, "red")]:
        _k13 = int(np.argmax(_var_cum >= _t) + 1)
        _axes[1].axhline(_t, color=_col, ls="--", label=f"{_t:.0%} → k = {_k13}")
        _axes[1].axvline(_k13, color=_col, ls=":", alpha=0.5)
    _axes[1].set_title("Variance cumulée")
    _axes[1].set_xlabel("Nombre de composantes")
    _axes[1].legend()
    _axes[1].grid(True, alpha=0.3)
    plt.tight_layout()

    _ref13 = np.cumsum(_var_ratio)
    _ok13 = (np.ndim(_var_cum) == 1 and len(_var_cum) == len(_ref13)
             and np.allclose(_var_cum, _ref13))
    _verdict13 = mo.callout(
        mo.md("✅ Code correct — `_var_cum` est bien la variance cumulée."
              if _ok13 else "❌ Pas encore."),
        kind="success" if _ok13 else "danger",
    )

    print(f"PC1 + PC2 → {_var_cum[1]:.1%} de la variance")
    print(f"80 % de variance → {int(np.argmax(_var_cum >= 0.80) + 1)} composantes")
    print(f"90 % de variance → {int(np.argmax(_var_cum >= 0.90) + 1)} composantes")
    mo.vstack([_verdict13, _fig13])
    return


@app.cell
def _(mo):
    mo.md("""
    Réponses aux questions :
    PC1+PC2 capturent environ 63,2 % de la variance. Il faut
    5 composantes pour atteindre 80 % et 7 pour atteindre 90 %,
    contre 30 variables de départ : l'ACP compresse fortement, ce qui
    est cohérent avec la corrélation moyenne élevée (≈0,39) observée
    dans l'EDA (variables redondantes → peu de "vraie" information
    indépendante).
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ✏️ Exercice 14 — Voir les biopsies en 2D

    `_Z` = `PCA(n_components=2).fit_transform(X_std)`
    """)
    return


@app.cell
def _(PCA, X_std, mo, np, plt, y):
    _Z = PCA(n_components=2).fit_transform(X_std)

    _fig14, _ax14 = plt.subplots(figsize=(7, 6))
    for _lab, _val, _colr in [("Maligne", 0, "red"), ("Bénigne", 1, "green")]:
        _m = y == _val
        _ax14.scatter(_Z[_m, 0], _Z[_m, 1], c=_colr, label=_lab, alpha=0.5, s=25)
    _ax14.set_xlabel("PC1")
    _ax14.set_ylabel("PC2")
    _ax14.set_title("Biopsies projetées par l'ACP (2 composantes)")
    _ax14.legend()
    _ax14.grid(True, alpha=0.3)

    _Z_ref = PCA(n_components=2).fit_transform(X_std)
    _ok14 = (np.shape(_Z) == _Z_ref.shape
             and np.allclose(np.abs(_Z), np.abs(_Z_ref), atol=1e-6))
    _verdict14 = mo.callout(
        mo.md("✅ Code correct — `_Z` est bien la projection sur 2 composantes."
              if _ok14 else "❌ Pas encore."),
        kind="success" if _ok14 else "danger",
    )
    mo.vstack([_verdict14, _fig14])
    return


@app.cell
def _(mo):
    mo.md("""
    Réponses aux questions :
    Les tumeurs malignes et bénignes forment deux groupes assez bien
    séparés, principalement le long de PC1 (les malignes ont des
    valeurs de PC1 plus élevées). Bien que PC1+PC2 ne capturent
    "que" ~63 % de la variance totale, cette variance concentre
    justement l'information la plus discriminante (taille/forme du
    noyau) : la séparation visuelle reste donc très nette.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ✏️ Exercice 15 — Que « signifient » les composantes ?

    `_load` = `(_pca.components_ * np.sqrt(_pca.explained_variance_[:, np.newaxis])).T`
    """)
    return


@app.cell
def _(PCA, X_std, feat_names, mo, np, plt, sns):
    _pca = PCA(n_components=2).fit(X_std)

    _load = (_pca.components_ * np.sqrt(_pca.explained_variance_[:, np.newaxis])).T

    _fig15, _ax15 = plt.subplots(figsize=(6, 10))
    sns.heatmap(np.round(_load, 2), annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=_ax15,
                xticklabels=["PC1", "PC2"], yticklabels=feat_names, linewidths=0.5)
    _ax15.set_title("Loadings — contribution des variables")
    _ax15.tick_params(axis="y", labelsize=7)

    _ref15 = (_pca.components_ * np.sqrt(_pca.explained_variance_[:, np.newaxis])).T
    _ok15 = (np.shape(_load) == _ref15.shape
             and np.allclose(np.abs(_load), np.abs(_ref15), atol=1e-6))
    _verdict15 = mo.callout(
        mo.md("✅ Code correct — `_load` est bien la matrice des loadings (30 × 2)."
              if _ok15 else "❌ Pas encore."),
        kind="success" if _ok15 else "danger",
    )

    _top = np.argsort(np.abs(_load[:, 0]))[::-1][:3]
    print("Les 3 variables les plus contributives à PC1 :")
    for _j in _top:
        print(f"  {feat_names[_j]:<28} loading = {_load[_j, 0]:+.2f}")
    mo.vstack([_verdict15, _fig15])
    return


@app.cell
def _(mo):
    mo.md("""
    Réponses aux questions :
    Les 3 variables les plus contributives à PC1 sont *mean concave
    points*, *mean concavity* et *worst concave points* : elles
    partagent toutes le même thème, la concavité/irrégularité du
    contour du noyau (une notion étroitement liée à la taille et à
    l'irrégularité globale de la tumeur, elle-même corrélée au rayon
    et au périmètre). On peut donc nommer PC1 « indice
    d'irrégularité/concavité du noyau ». Cliniquement, une valeur
    élevée de PC1 correspond à des noyaux plus irréguliers et
    concaves, ce qui est associé à la malignité.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ✏️ Exercice 16 — L'ACP aide-t-elle à diagnostiquer ?

    `_k` = nombre de composantes ACP à tester (essayer 2, puis 7 pour 90 %).
    """)
    return


@app.cell
def _(
    LogisticRegression,
    PCA,
    Pipeline,
    StandardScaler,
    X,
    cross_val_score,
    mo,
    y,
):
    _k = 2

    if not (isinstance(_k, int) and 1 <= _k <= 30):
        _out16 = mo.callout(
            mo.md("❌ `_k` invalide — choisissez un entier entre 1 et 30."),
            kind="danger",
        )
    else:
        _pipe_full = Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000)),
        ])
        _pipe_pca = Pipeline([
            ("sc", StandardScaler()),
            ("pca", PCA(n_components=_k)),
            ("clf", LogisticRegression(max_iter=5000)),
        ])
        _s_full = cross_val_score(_pipe_full, X, y, cv=5, scoring="accuracy")
        _s_pca = cross_val_score(_pipe_pca, X, y, cv=5, scoring="accuracy")

        _txt = (
            f"Seuil naïf (tout bénin)    : {max((y == 1).mean(), (y == 0).mean()):.3f}\n"
            f"Sans ACP  (30 variables)   : {_s_full.mean():.3f} ± {_s_full.std():.3f}\n"
            f"Avec ACP  ({_k:>2} composantes) : {_s_pca.mean():.3f} ± {_s_pca.std():.3f}"
        )
        print(_txt)
        _verdict16 = mo.callout(
            mo.md(f"✅ Paramètre valide — comparaison lancée avec k = {_k} composantes."),
            kind="success",
        )
        _out16 = mo.vstack([_verdict16, mo.md(f"```\n{_txt}\n```")])
    _out16
    return


@app.cell
def _(mo):
    mo.md("""
    Réponses aux questions :
    Avec seulement k = 2 composantes (au lieu de 30), la perte
    d'accuracy est faible : le modèle avec ACP reste très proche du
    modèle sur les 30 variables. Le seuil naïf « tout bénin » vaut
    ~63 % (proportion de bénignes) ; les deux modèles le battent
    très largement (accuracy autour de 95-98 %), ce qui confirme que
    l'ACP conserve l'essentiel de l'information utile au diagnostic.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    🧠 Questions de synthèse

    1. Oui, l'ACP résume efficacement les 30 variables en 2-3
       indicateurs utiles, car la corrélation absolue moyenne mesurée
       dans l'EDA est élevée : les variables sont très redondantes,
       donc peu de composantes suffisent à capturer l'essentiel de
       l'information.

    2. Rayon, périmètre et aire d'un noyau sont géométriquement liés
       (périmètre ≈ 2πr, aire ≈ πr²) : ce sont presque la même
       information exprimée différemment. Cette redondance est
       exactement ce que l'ACP exploite pour compresser les données.

    3. Non, l'ACP est non supervisée : elle n'utilise jamais les
       étiquettes de diagnostic. La séparation observée sur PC1
       émerge naturellement parce que la source de variance dominante
       (taille/irrégularité du noyau) est corrélée avec la malignité.
       Pour maximiser explicitement la séparation entre classes,
       il faudrait utiliser une méthode supervisée comme la LDA
       (analyse discriminante linéaire), qui utilise y.

    4. Non : avec des variables peu corrélées (chacune apportant une
       information distincte), la variance serait répartie sur
       beaucoup plus d'axes, et il faudrait garder presque toutes
       les composantes pour atteindre 80-90 % de variance expliquée.
       L'ACP compresserait donc beaucoup moins bien.
    """)
    return


if __name__ == "__main__":
    app.run()
