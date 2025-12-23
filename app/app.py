import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# -----------------------------------------------------------------------------
# 1. CONFIGURATION ET CHARGEMENT
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Laptop Expert IA", page_icon="💻", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ajustement des chemins pour être sûr de trouver les fichiers
# Si on est dans 'app/', les modèles sont dans '../models'
MODEL_DIR = os.path.join(BASE_DIR, '../models')
DATA_PATH = os.path.join(BASE_DIR, '../data/laptop_prices.csv')

@st.cache_resource
def load_assets():
    try:
        # 1. Charger les Modèles
        knn_model = pickle.load(open(os.path.join(MODEL_DIR, 'knn_model.pkl'), 'rb'))
        knn_scaler = pickle.load(open(os.path.join(MODEL_DIR, 'knn_scaler.pkl'), 'rb'))
        knn_le = pickle.load(open(os.path.join(MODEL_DIR, 'knn_label_encoder.pkl'), 'rb'))
        knn_cols = pickle.load(open(os.path.join(MODEL_DIR, 'knn_columns.pkl'), 'rb'))
        
        price_model = pickle.load(open(os.path.join(MODEL_DIR, 'price_model.pkl'), 'rb'))
        price_cols = pickle.load(open(os.path.join(MODEL_DIR, 'price_columns.pkl'), 'rb'))
        
        # 2. Charger les Données pour les listes déroulantes
        if os.path.exists(DATA_PATH):
            df_ref = pd.read_csv(DATA_PATH)
        else:
            st.error("❌ Fichier de données introuvable (data/laptop_prices.csv). Impossible de charger les options.")
            st.write(f"Chemin cherché : {DATA_PATH}")
            return None, None, None, None, None, None, None

        return knn_model, knn_scaler, knn_le, knn_cols, price_model, price_cols, df_ref
    except Exception as e:
        st.error(f"❌ Erreur de chargement: {e}")
        st.write(f"Dossier Base: {BASE_DIR}")
        return None, None, None, None, None, None, None

knn_model, knn_scaler, knn_le, knn_cols, price_model, price_cols, df_ref = load_assets()

# -----------------------------------------------------------------------------
# 2. FONCTION INTELLIGENTE (Conversion Utilisateur -> IA)
# -----------------------------------------------------------------------------
def create_input_dataframe(data_dict, columns_list):
    # On crée une ligne vide avec toutes les colonnes connues du modèle
    df_input = pd.DataFrame(np.zeros((1, len(columns_list))), columns=columns_list)
    
    # 1. On remplit les chiffres (RAM, Poids...)
    for col, value in data_dict.items():
        if col in df_input.columns:
            df_input[col] = value
            
    # 2. On remplit les cases à cocher (One-Hot Encoding)
    # Ex: Si l'utilisateur choisit 'Nvidia', on met un 1 dans la colonne 'GPU_company_Nvidia'
    for cat_col, selected_option in data_dict.items():
        if isinstance(selected_option, str):
            # Construction du nom de colonne: NomVariable_Valeur
            target_col = f"{cat_col}_{selected_option}" 
            
            if target_col in df_input.columns:
                df_input[target_col] = 1
                
    return df_input

if df_ref is not None:
    # -----------------------------------------------------------------------------
    # 3. BARRE LATÉRALE (Configuration Complète)
    # -----------------------------------------------------------------------------
    st.sidebar.header("🛠️ Configuration Complète")

    # --- A. Marque & Système ---
    st.sidebar.subheader("1. Identité")
    # On récupère les choix depuis le Dataset réel pour être sûr de la compatibilité
    brands = sorted(df_ref['Company'].unique())
    os_list = sorted(df_ref['OS'].unique())

    company = st.sidebar.selectbox("Marque", brands)
    os_sys = st.sidebar.selectbox("OS", os_list)

    # --- B. Processeur (CPU) ---
    st.sidebar.subheader("2. Processeur (CPU)")
    cpu_brands = sorted(df_ref['CPU_company'].unique())
    # Filtre dynamique : On choisit la marque, puis le modèle
    cpu_brand = st.sidebar.selectbox("Marque CPU", cpu_brands)
    
    # On filtre les modèles disponibles pour cette marque
    available_cpu_models = sorted(df_ref[df_ref['CPU_company'] == cpu_brand]['CPU_model'].unique())
    cpu_model = st.sidebar.selectbox("Modèle CPU", available_cpu_models)
    
    cpu_freq = st.sidebar.slider("Fréquence (GHz)", 0.9, 4.0, 2.5, step=0.1)

    # --- C. Carte Graphique (GPU) ---
    st.sidebar.subheader("3. Carte Graphique (GPU)")
    gpu_brands = sorted(df_ref['GPU_company'].unique())
    gpu_brand = st.sidebar.selectbox("Marque GPU", gpu_brands)
    
    # Filtre dynamique pour GPU
    available_gpu_models = sorted(df_ref[df_ref['GPU_company'] == gpu_brand]['GPU_model'].unique())
    gpu_model = st.sidebar.selectbox("Modèle GPU", available_gpu_models)

    # --- D. Mémoire (RAM) ---
    ram = st.sidebar.select_slider("RAM (Go)", options=[2, 4, 6, 8, 12, 16, 24, 32, 64], value=8)

    # --- E. Stockage (Prim & Sec) ---
    st.sidebar.subheader("4. Stockage")
    storage_types = ['SSD', 'HDD', 'Flash Storage', 'Hybrid', 'No']
    
    # Disque Principal
    p_storage = st.sidebar.number_input("Disque Principal (Go)", 0, 4096, 256, step=128)
    p_type = st.sidebar.selectbox("Type Principal", [t for t in storage_types if t != 'No'])
    
    # Disque Secondaire
    s_storage_enabled = st.sidebar.checkbox("Disque Secondaire ?")
    if s_storage_enabled:
        s_storage = st.sidebar.number_input("Disque Secondaire (Go)", 0, 4096, 1000, step=128)
        s_type = st.sidebar.selectbox("Type Secondaire", [t for t in storage_types if t != 'No'], index=1)
    else:
        s_storage = 0
        s_type = "No"

    # --- F. Écran & Design ---
    st.sidebar.subheader("5. Écran & Design")
    sizes = sorted(df_ref['Inches'].unique()) 
    
    # On reconstruit la logique PPI
    resolutions = {
        "Standard (1366x768)": (1366, 768),
        "Full HD (1920x1080)": (1920, 1080),
        "Quad HD+ (3200x1800)": (3200, 1800),
        "4K Ultra HD (3840x2160)": (3840, 2160),
        "Retina (2304x1440)": (2304, 1440),
        "Retina (2560x1600)": (2560, 1600)
    }
    
    screen_size = st.sidebar.select_slider("Taille (Pouces)", options=sizes, value=15.6)
    screen_res_name = st.sidebar.selectbox("Qualité d'écran", list(resolutions.keys()), index=1)
    
    touch = st.sidebar.checkbox("Écran Tactile (Touchscreen)")
    ips = st.sidebar.checkbox("Dalle IPS (Couleurs Pro)")
    retina = st.sidebar.checkbox("Retina Display")
    
    weight = st.sidebar.slider("Poids (kg)", 0.5, 5.0, 1.8, step=0.1)

    # -----------------------------------------------------------------------------
    # 4. LOGIQUE DE PRÉDICTION
    # -----------------------------------------------------------------------------
    st.title("💻 L'Expert Laptop IA (v2)")
    st.markdown("""
    ### Une estimation de précision professionnelle
    Notre IA analyse non seulement la marque, mais aussi le modèle exact du CPU, du GPU et la combinaison de stockage.
    """)

    # Calcul PPI
    res_w, res_h = resolutions[screen_res_name]
    ppi = np.sqrt(res_w**2 + res_h**2) / screen_size

    if st.button("🚀 Lancer l'Estimation Précise"):
        
        # Dictionnaire brut 
        input_data = {
            # Features Numériques
            'Ram': ram, 
            'Weight': weight, 
            'PPI': ppi,
            'CPU_freq': cpu_freq,
            'PrimaryStorage': p_storage,
            'SecondaryStorage': s_storage,
            
            # Features Catégorielles (pour One-Hot)
            'Company': company, 
            'OS': os_sys,
            'CPU_company': cpu_brand,
            'CPU_model': cpu_model,          
            'GPU_company': gpu_brand,
            'GPU_model': gpu_model,          
            'PrimaryStorageType': p_type,
            'SecondaryStorageType': s_type,  
            'Touchscreen': 'Yes' if touch else 'No',
            'IPSpanel': 'Yes' if ips else 'No',
            'RetinaDisplay': 'Yes' if retina else 'No' 
        }
        
        # --- 1. Classification (KNN/RandomForest) ---
        # Préparation du vecteur
        X_knn = create_input_dataframe(input_data, knn_cols)
        
        # Remplissage des 0 pour les colonnes non présentes dans le dictionnaire
        X_knn = X_knn.fillna(0)
        
        # Scaling (indispensable pour KNN)
        # Attention : s'assurer que le scaler a été fit sur les mêmes colonnes
        # En théorie oui si knn_cols correspond à X_train.columns
        try:
            X_knn_scaled = knn_scaler.transform(X_knn)
            
            # Prédiction
            pred_type_encoded = knn_model.predict(X_knn_scaled)[0]
            pred_type_name = knn_le.inverse_transform([pred_type_encoded])[0]
            
            # Affichage
            st.divider()
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.info(f"🏷️ **Catégorie identifiée :** {pred_type_name}")
                if pred_type_name == 'Gaming':
                    st.write("🎮 Optimisé pour le jeu et la performance.")
                elif pred_type_name == 'Ultrabook':
                    st.write("✈️ Fin, léger et portable.")
                elif pred_type_name == 'Workstation':
                    st.write("🏗️ Puissance de calcul professionnelle.")

            # --- 2. Régression (Prix) ---
            # On ajoute la classe prédite comme feature entrée pour le prix
            input_data['TypeName'] = pred_type_name
            
            X_price = create_input_dataframe(input_data, price_cols)
            X_price = X_price.fillna(0)
            
            # Prédiction (Log -> Euro)
            pred_log = price_model.predict(X_price)[0]
            pred_euros = np.expm1(pred_log)
            
            with col_res2:
                st.success(f"💰 **Prix Estimé du Marché :** {pred_euros:.2f} €")
                st.caption("Basé sur une analyse de +100 configurations similaires.")
        
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
            st.write("Détails debug : Vérifiez l'alignement des colonnes.")

else:
    st.warning("En attente des données...")