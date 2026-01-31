# 💻 Laptop Expert AI

**Projet de Machine Learning & Application Web pour l'estimation de prix et la classification d'ordinateurs portables.**

🔗 **Démo en ligne :** [Accéder à l'application](https://laptopexpert-hw7h2dv9gv9otssw7e74hn.streamlit.app/)

Ce projet a été réalisé dans le cadre du module de **Machine Learning (M1 IA)**. Il propose une solution complète (Notebooks d'analyse + Application Web) capable de :
1.  **Classifier** un ordinateur portable selon sa configuration (Gaming, Ultrabook, Workstation...).
2.  **Estimer** son prix de marché exact en Euros.

---

## 🚀 Fonctionnalités

*   **Interface Web Moderne** : Une application **Streamlit** interactive et facile à utiliser.
*   **Classification Intelligente** : Un modèle **Random Forest** entraîné pour prédire la catégorie du laptop (ex: Ultrabook, Gaming) en fonction de ses specs.
*   **Estimation Précise du Prix** : Un modèle de **Régression Ridge** optimisé qui prédit le prix en fonction de plus de 10 critères (CPU, GPU, RAM, Stockage SSD/HDD, résolution d'écran, etc.).
*   **Données Réelles** : L'application charge dynamiquement les modèles de processeurs et cartes graphiques existants sur le marché pour des choix précis.
*   **Contexte Temporel** : Les modèles ont été entraînés sur des données de **2017 et début 2018**, reflétant les prix du marché de cette période.

## 🛠️ Stack Technique

*   **Langage :** Python 3.9+
*   **Interface Utilisateur :** [Streamlit](https://streamlit.io/)
*   **Machine Learning :** [Scikit-Learn](https://scikit-learn.org/) (KNN, Linear Regression, Ridge, SMOTE)
*   **Manipulation de Données :** Pandas, NumPy
*   **Visualisation :** Matplotlib, Seaborn

## 📂 Structure du Projet

```bash
Laptop_Expert/
├── app/
│   └── app.py               # 🚀 Le script principal de l'application Web
├── data/
│   └── laptop_prices.csv    # 📊 Le jeu de données utilisé (Source : [Kaggle](https://www.kaggle.com/datasets/owm4096/laptop-prices/data))
├── models/                  # 🧠 Les modèles IA entraînés (.pkl)
│   ├── knn_model.pkl        # Modèle de classification
│   ├── price_model.pkl      # Modèle de régression
│   └── ...                  # Scalers et encodeurs pour le prétraitement
├── notebooks/               # 📓 Les carnets d'expérimentation
│   ├── Classification_Notebook.ipynb
│   └── Regression_Notebook.ipynb
└── README.md                # 📄 Ce fichier
```

## 💿 Installation et Lancement

1.  **Cloner le dépôt**
    ```bash
    git clone https://github.com/Sofiane-Meziane/Laptop_Expert.git
    cd Laptop_Expert
    ```

2.  **Créer un environnement virtuel (Recommandé)**
    Il est pratique de créer un environnement propre au projet pour éviter les conflits de versions.
    
    *   **Windows :**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   **Mac/Linux :**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Installer les dépendances**
    Une fois l'environnement activé, installez les paquets nécessaires listés dans `requirements.txt` :
    ```bash
    pip install -r requirements.txt
    ```

4.  **Lancer l'application**
    ```bash
    streamlit run app/app.py
    ```
    Une page web s'ouvrira automatiquement dans votre navigateur (généralement sur `http://localhost:8501`).

## 🧠 Détails des Modèles

### 1. Classification (Notebook 1)
*   **Objectif :** Prédire le `TypeName` (Ultrabook, Gaming, Notebook...).
*   **Méthode :** Random Forest Classifier.
*   **Performance :** Ce modèle surpasse le KNN avec une précision globale de **84.31%**.
*   **Optimisation :** Utilisation de **SMOTE** pour équilibrer les classes minoritaires.

### 2. Régression (Notebook 2)
*   **Objectif :** Prédire le `Price_euros`.
*   **Méthode :** Régression Ridge.
*   **Performance :** Le modèle atteint un score $R^2$ d’environ **0.86** sur le jeu de test.  
    L’hyperparamètre de régularisation est optimisé par **validation croisée (5-fold CV)** afin d’améliorer la capacité de généralisation du modèle et d’éviter toute fuite d’information.  
    L’apprentissage est réalisé sur le logarithme du prix pour une meilleure robustesse face aux valeurs extrêmes.
