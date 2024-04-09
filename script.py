from scipy.stats import shapiro
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def afficher_types_attributs(df):
    print(df.dtypes)


def valeursManquantes(df):
    
    valeurs_manquantes = df.isnull().sum()
    print("Valeurs manquantes par colonne pour ")
    print(valeurs_manquantes)
    return valeurs_manquantes



def traiter_valeurs_aberrantes(df, col):
    df[col + '_zscore'] = zscore(df[col].dropna())
    df = df[(df[col + '_zscore'].abs() <= 3)]
    df.drop(col + '_zscore', axis=1, inplace=True)
    return df

def nettoyage_lots(df):
    # Suppression des doublons
    df.drop_duplicates(inplace=True)

    # Imputation des valeurs manquantes pour les colonnes numériques avec la médiane
    median_columns = ['numberTenders', 'contractDuration', 'publicityDuration', 'awardPrice', 'awardEstimatedPrice']
    for col in median_columns:
        if col in df.columns:
            median = df[col].median()
            df[col].fillna(median, inplace=True)

    # Calcul du mode pour 'awardDate' et imputation des valeurs manquantes avec le mode
    if 'awardDate' in df.columns:
        awardDate_mode = df['awardDate'].mode()[0]
        df['awardDate'].fillna(awardDate_mode, inplace=True)

    # Imputation pour les colonnes catégorielles/booléennes avec le mode ou 0
    mode_or_zero_columns = ['onBehalf', 'jointProcurement', 'fraAgreement', 'fraEstimated', 
                            'accelerated', 'outOfDirectives', 'contractorSme', 'numberTendersSme', 
                            'gpa', 'multipleCae', 'typeOfContract', 'topType', 'subContracted', 'renewal']
    for col in mode_or_zero_columns:
        if col in df.columns:
            mode = df[col].mode()[0] if pd.notnull(df[col].mode()[0]) else 0
            df[col].fillna(mode, inplace=True)

    # Suppression de la colonne 'lotsNumber' si elle existe
    if 'lotsNumber' in df.columns:
        df.drop('lotsNumber', axis=1, inplace=True)

    # Détection et traitement des valeurs aberrantes pour les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df.columns:
            df = traiter_valeurs_aberrantes(df, col)
    valeurs_manquantes = df.isnull().sum()
    print("Valeurs manquantes apres neteoyage par colonne pou LOT ")
    print(valeurs_manquantes)

    return df

# Fonction pour analyser et visualiser les variables numériques
def analyze_and_plot_variable(df, column_name):
    # Initialiser mean et std à None
    mean, std = None, None
    
    # Suppression des valeurs NaN pour l'analyse
    data = df[column_name].dropna()

    # Affichage des statistiques descriptives
    print(data.describe())

    # Visualisation de la distribution sans transformation logarithmique
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, bins=50)
    plt.title(f'Distribution of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.show()

    # Effectuer le test de Shapiro-Wilk pour la normalité
    if len(data) > 5000:  # Le test de Shapiro-Wilk a une limite de taille d'échantillon
        data = data.sample(5000)
    shapiro_stat, shapiro_p = shapiro(data)
    print(f'Shapiro-Wilk Test: Statistics={shapiro_stat}, p-value={shapiro_p}\n')

    # Calculer la moyenne et l'écart-type pour les données originales
    mean, std = np.mean(data), np.std(data)
    
    # Vérification de la possibilité d'appliquer la transformation logarithmique
    non_positive_values = data[data < 0]
    if not non_positive_values.empty:
        print(f"Data contains non-positive values in column {column_name}:")
        print(non_positive_values)
    else:
        # Appliquer une transformation logarithmique pour une meilleure visualisation
        data_log = np.log(data)

        # Visualisation de la distribution après transformation log
        plt.figure(figsize=(10, 6))
        sns.histplot(data_log, kde=True, bins=50)
        plt.title(f'Logarithmic Distribution of {column_name}')
        plt.xlabel(f'Log of {column_name}')
        plt.ylabel('Frequency')
        plt.show()

        # Effectuer le test de Shapiro-Wilk pour la normalité sur les données transformées
        if len(data_log) > 5000:
            data_log = data_log.sample(5000)
        shapiro_stat_log, shapiro_p_log = shapiro(data_log)
        print(f'Log-Transformed Shapiro-Wilk Test: Statistics={shapiro_stat_log}, p-value={shapiro_p_log}\n')
        
        # Mettre à jour mean et std pour les données transformées logarithmiquement
        mean, std = np.mean(data_log), np.std(data_log)

    return mean, std  # Retourner la moyenne et l'écart-type

# r analyzing categorical variables
def analyze_and_plot_categorical_variable(df, column_name, max_categories=20):
    # Calcul des fréquences des catégories
    value_counts = df[column_name].value_counts()

    # Si max_categories est 0, afficher toutes les catégories
    if max_categories == 0:
        max_categories = len(value_counts)

    # Déterminer le nombre de catégories à afficher pour les "top" et "bottom" catégories
    num_categories = min(max_categories, len(value_counts))
    
    # Visualisation des 'top' catégories les plus fréquentes
    plt.figure(figsize=(10, min(10, 0.5 * num_categories)))
    sns.barplot(y=value_counts.index[:num_categories], x=value_counts.values[:num_categories])
    plt.title(f'Top {num_categories} Most Frequent {column_name}')
    plt.xlabel('Count')
    plt.ylabel(column_name)
    plt.show()

    # Visualisation des 'bottom' catégories les moins fréquentes si le nombre total de catégories est supérieur à max_categories
    if len(value_counts) > max_categories:
        plt.figure(figsize=(10, min(10, 0.5 * num_categories)))
        sns.barplot(y=value_counts.index[-num_categories:], x=value_counts.values[-num_categories:])
        plt.title(f'Top {num_categories} Least Frequent {column_name}')
        plt.xlabel('Count')
        plt.ylabel(column_name)
        plt.show()


# Update the function to iterate over all files and columns to include categorical analysis
def analyze_datasets(files):
    for file_path, columns in files.items():
        print(f"Analyzing {file_path}")
        df = pd.read_csv(file_path)

        for column in columns:
            # Check if the column is an identifier or code
            if 'Id' in column or 'Code' in column or 'SIRET' in column:
                print(f"Processing categorical ID column: {column}")
                analyze_and_plot_id_variable(df, column)
            elif pd.api.types.is_numeric_dtype(df[column]):
                print(f"Processing numerical column: {column}")
                mean, std = analyze_and_plot_variable(df, column)
                print(f"Mean: {mean}, Std Deviation: {std}\n")
            elif pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column]):
                print(f"Processing categorical column: {column}")
                analyze_and_plot_categorical_variable(df, column)
            else:
                print(f"Column {column} is skipped from analysis.\n")


def analyze_and_plot_id_variable(df, column_name):
    value_counts = df[column_name].value_counts()[:20]  # Ne prendre que les 20 premiers ID pour la lisibilité

    plt.figure(figsize=(10, 6))
    sns.barplot(x=value_counts.index, y=value_counts.values)
    plt.title(f'Distribution of {column_name} - Top 20 IDs')
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.show()




def processus_analyse_donnees(files_to_analyze,fileColumns):
    for fichier in files_to_analyze:
        df = pd.read_csv(fichier)
        
        # Afficher les types d'attributs et les valeurs manquantes
        afficher_types_attributs(df)
        #Affichage des valeurs manquantes
        valeursManquantes(df)
        
        # Nettoyage spécifique pour le fichier Lots.csv
        if 'Lots.csv' in fichier:
            valeursManquantes(df)
            df = nettoyage_lots(df)
            #Visualisation de la distribution de log_awardPrice par typeOfContract
            visualiser_association_categorie_numerique(df, 'gpa', 'awardPrice')
            visualiser_association_categorie_numerique(df,"accelerated","awardPrice")
            visualiser_association_categorie_numerique(df,"typeOfContract","awardPrice")
            df_sample = df.sample(n=1000, random_state=1)
            visualiser_association(df_sample,"contractDuration","awardPrice")
    analyze_datasets(fileColumns)


def visualiser_association(df, variable_x, variable_y):
    # Remplacez les valeurs 0 par NaN pour éviter les erreurs de log(0)
    df[variable_y] = df[variable_y].replace(0, np.nan)
    df.dropna(subset=[variable_y], inplace=True)

    # Appliquez la transformation log à la variable y si nécessaire
    df['log_' + variable_y] = np.log(df[variable_y])

    sns.jointplot(x=variable_x, y='log_' + variable_y, data=df, kind="reg")
    plt.show()

def calculer_correlation(df, variable1, variable2):
    # Appliquez la transformation log à la variable y si nécessaire
    df['log_' + variable2] = np.log(df[variable2].replace(0, np.nan))
    df.dropna(subset=['log_' + variable2], inplace=True)

    corr = df[[variable1, 'log_' + variable2]].corr().iloc[0, 1]
    print(f"Corrélation entre {variable1} et log de {variable2}: {corr}")

def visualiser_association_categorie_numerique(df, variable_cat, variable_num):
    df['log_' + variable_num] = np.log(df[variable_num] + 1)  # On ajoute 1 pour éviter log(0)
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=variable_cat, y='log_' + variable_num, data=df)
    plt.xticks(rotation=45)
    plt.title(f"Distribution de log de {variable_num} par {variable_cat}")
    plt.show()
    df_lots['log_awardPrice'] = np.log(df_lots['awardPrice'] + 1)  # Ajoutez 1 pour éviter le log de 0

    # Groupez par typeOfContract et calculez les statistiques descriptives pour log_awardPrice
    grouped_stats = df_lots.groupby('typeOfContract')['log_awardPrice'].describe()

    print(grouped_stats)



def clustering_and_visualization(filepath_sirene_extended):
    # Chargement des données enrichies
    df = pd.read_csv(filepath_sirene_extended)
    
    # Convertissez la date de création en âge de l'entreprise
    df['dateCreationUniteLegale'] = pd.to_datetime(df['dateCreationUniteLegale'], errors='coerce')
    df['ageUniteLegale'] = (pd.Timestamp.now() - df['dateCreationUniteLegale']).dt.days / 365.25
    
    # Sélectionnez les variables pour le clustering
    features = df[['activitePrincipaleUniteLegale', 'trancheEffectifsUniteLegale', 'categorieEntreprise', 'ageUniteLegale']].copy()
    
    # Prétraitement pour les valeurs manquantes et encodage des variables catégorielles
    features['ageUniteLegale'] = SimpleImputer(strategy='median').fit_transform(features[['ageUniteLegale']])
    features = pd.get_dummies(features, columns=['activitePrincipaleUniteLegale', 'trancheEffectifsUniteLegale', 'categorieEntreprise'])
    
    # Normalisation des données
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Application de KMeans
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['cluster'] = kmeans.fit_predict(features_scaled)
    
    # Réduction de la dimensionnalité pour la visualisation
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features_scaled)
    
    # Création d'un DataFrame pour la visualisation
    df_vis = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    df_vis['cluster'] = df['cluster']

 
    # Visualisation des clusters
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=df_vis, palette='viridis', alpha=0.7)
    plt.title('Visualisation des clusters')
    plt.show()
    
    # Analyse des clusters
    print(df.groupby('cluster').mean())  # Cela imprimera les moyennes des variables pour chaque cluster
    

    

    # Afficher les centroïdes des clusters
    print("Centroids of clusters:")
    print(pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features.columns))


    return df


def analyze_temporal_trends(filepath):
    # Charger les données
    df = pd.read_csv(filepath)
    
    # Convertir les dates en datetime
    df['awardDate'] = pd.to_datetime(df['awardDate'], errors='coerce')
    
    # Extraire l'année et le mois de la date d'attribution
    df['year'] = df['awardDate'].dt.year
    df['month'] = df['awardDate'].dt.month
    
    # Analyser le nombre de contrats attribués par mois
    contracts_per_month = df.groupby(['year', 'month'])['lotId'].count().reset_index(name='count')
    
    # Visualiser les tendances
    plt.figure(figsize=(15, 6))
    sns.lineplot(data=contracts_per_month, x='month', y='count', hue='year', marker='o')
    plt.title('Nombre de contrats attribués par mois')
    plt.xlabel('Mois')
    plt.ylabel('Nombre de contrats')
    plt.legend(title='Année')
    plt.show()
    
    return contracts_per_month



def afficher_types_attributs(df):
    print(df.dtypes)


def traiter_valeurs_aberrantes_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]


def traiter_valeurs_aberrantes(df, col):
    df[col + '_zscore'] = zscore(df[col].dropna())
    df = df[(df[col + '_zscore'].abs() <= 3)]
    df.drop(col + '_zscore', axis=1, inplace=True)
    return df


def nettoyage_lots(df):
    # Suppression des doublons
    df.drop_duplicates(inplace=True)

    # Suppression des lignes où awardPrice est égal à 0
    df = df[(df['awardPrice'] != 0) & (df['awardPrice'] < 8.36e+13)]
    df = df[df['awardPrice'] >= 100]
    # Imputation des valeurs manquantes pour les colonnes numériques avec la médiane
    median_columns = ['numberTenders', 'contractDuration', 'publicityDuration', 'awardPrice', 'awardEstimatedPrice']
    for col in median_columns:
        if col in df.columns:
            median = df[col].median()
            df[col].fillna(median, inplace=True)

    # Calcul du mode pour 'awardDate' et imputation des valeurs manquantes avec le mode
    if 'awardDate' in df.columns:
        awardDate_mode = df['awardDate'].mode()[0]
        df['awardDate'].fillna(awardDate_mode, inplace=True)

    # Imputation pour les colonnes catégorielles/booléennes avec le mode ou 0
    mode_or_zero_columns = ['onBehalf', 'jointProcurement', 'fraAgreement', 'fraEstimated',
                            'accelerated', 'outOfDirectives', 'contractorSme', 'numberTendersSme',
                            'gpa', 'multipleCae', 'typeOfContract', 'topType', 'subContracted', 'renewal']
    for col in mode_or_zero_columns:
        if col in df.columns:
            mode = df[col].mode()[0] if pd.notnull(df[col].mode()[0]) else 0
            df[col].fillna(mode, inplace=True)

    # Suppression de la colonne 'lotsNumber' si elle existe
    if 'lotsNumber' in df.columns:
        df.drop('lotsNumber', axis=1, inplace=True)

    # Détection et traitement des valeurs aberrantes pour les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df.columns:
            df = traiter_valeurs_aberrantes(df, col)
    df = traiter_valeurs_aberrantes_iqr(df, 'awardPrice')
    return df

df_lots = pd.read_csv("data/Lots.csv")

df_lots = nettoyage_lots(df_lots)


"""

-------- Quelles sont les caractéristiques des lots qui influencent le plus leur prix d’attribution
(awardPrice) ? --------

"""

def linear_regression_analysis(df):
    # Sélectionner les caractéristiques et la cible
    X = df[['numberTenders', 'contractDuration', 'publicityDuration', 'awardEstimatedPrice']]  # Ajustez les caractéristiques
    y = df['awardPrice']  # La variable cible

    # Diviser les données en ensembles de formation et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normaliser les caractéristiques
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entraîner le modèle de régression linéaire
    regressor = LinearRegression()
    regressor.fit(X_train_scaled, y_train)

    # Évaluer le modèle
    y_pred = regressor.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Examiner les coefficients
    coefficients = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
    print(coefficients)

    coefficients = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient']).sort_values(by='Coefficient')
    coefficients.plot(kind='bar', title='Influence des caractéristiques sur le prix d\'attribution')
    plt.show()



"""

-------- Quelles sont les caractéristiques des lots qui influencent le plus leur prix d’attribution
(awardPrice) ? --------

"""

def categorical_variable_analysis(df):
    # Encodage des variables catégorielles pour 'typeOfContract'
    label_encoder = LabelEncoder()
    df['typeOfContract_encoded'] = label_encoder.fit_transform(df['typeOfContract'])

    # Visualisation avec un boxplot pour voir la distribution de 'awardPrice' par 'typeOfContract'
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='typeOfContract', y='awardPrice', data=df)
    plt.title('Distribution des prix d\'attribution par type de contrat')
    plt.xlabel('Type de Contrat')
    plt.ylabel('Prix d\'Attribution')
    plt.xticks(rotation=45)  # Rotates the labels on the x-axis to make them readable
    plt.show()

    # Calcul de la corrélation
    correlation_matrix = df[['typeOfContract_encoded', 'awardPrice']].corr()
    print(correlation_matrix)

    # Visualisation de la matrice de corrélation
    sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu")
    plt.title('Heatmap de la Corrélation entre Type de Contrat et Prix d’Attribution')
    plt.show()


"""

-------- Quelles sont les principales caractéristiques qui définissent les profils des lots dans les marchés publics, 
et comment ces caractéristiques se regroupent-elles ? --------

"""

def acp_visualization(df):
    # Sélectionner les colonnes numériques pour l'ACP
    num_features = df.select_dtypes(include=['int64', 'float64']).drop(columns=['lotId', 'tedCanId', 'correctionsNb']).columns

    # Remplacer les valeurs manquantes par la médiane et standardiser les données
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    # Appliquer l'imputation et la normalisation
    df_num_imputed = imputer.fit_transform(df[num_features])
    df_num_scaled = scaler.fit_transform(df_num_imputed)

    # Appliquer l'ACP
    pca = PCA(n_components=2) # Vous pouvez ajuster le nombre de composants
    principal_components = pca.fit_transform(df_num_scaled)

    # Visualiser les deux premières composantes principales
    plt.figure(figsize=(8, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
    plt.xlabel('Première composante principale')
    plt.ylabel('Deuxième composante principale')
    plt.title('ACP des Lots des Marchés Publics')
    plt.show()

    # Afficher la variance expliquée par chaque composante principale
    print(pca.explained_variance_ratio_)


def analyser_correlation_distance_prix(lots_csv, buyers_csv, suppliers_csv, agents_csv):
    # Fonction Haversine pour calculer la distance
    def haversine(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Rayon de la Terre en kilomètres
        return c * r

    # Charger les données
    df_lots = pd.read_csv(lots_csv)
    df_buyers = pd.read_csv(buyers_csv)
    df_suppliers = pd.read_csv(suppliers_csv)
    df_agents = pd.read_csv(agents_csv)

    # Jointure pour récupérer les coordonnées des acheteurs et des fournisseurs
    df_buyers_coords = pd.merge(df_buyers, df_agents, on='agentId', how='left')
    df_suppliers_coords = pd.merge(df_suppliers, df_agents, on='agentId', how='left')

    # Fusionner ces données avec df_lots sur 'lotId'
    df_merged = pd.merge(df_lots, df_buyers_coords[['lotId', 'latitude', 'longitude']], on='lotId', how='left')
    df_merged = pd.merge(df_merged, df_suppliers_coords[['lotId', 'latitude', 'longitude']], on='lotId', how='left', suffixes=('', '_supplier'))

    # Calcul de la distance
    df_merged['distance_km'] = df_merged.apply(
        lambda row: haversine(
            row['longitude'], row['latitude'], 
            row['longitude_supplier'], row['latitude_supplier']
        ), 
        axis=1
    )

    # Analyser la corrélation entre la distance et le montant du contrat
    correlation = df_merged['distance_km'].corr(df_merged['awardPrice'])
    print(f"Corrélation entre la distance et le montant du contrat: {correlation}")

    # Visualisation
    plt.figure(figsize=(10,6))
    plt.scatter(df_merged['distance_km'], df_merged['awardPrice'], alpha=0.5)
    plt.xscale('log') # Utilisation d'une échelle logarithmique si nécessaire
    plt.yscale('log') # Utilisation d'une échelle logarithmique si nécessaire
    plt.xlabel('Distance (km)')
    plt.ylabel('Montant du Contrat (log scale)')
    plt.title('Distance vs Montant du Contrat')
    plt.grid(True)
    plt.show()

    return correlation, df_merged



def partie_questions():

    lots_csv = 'data/Lots.csv'
    buyers_csv = 'data/LotBuyers.csv'
    suppliers_csv = 'data/LotSuppliers.csv'
    agents_csv = 'data/Agents.csv'
    analyser_correlation_distance_prix(lots_csv, buyers_csv, suppliers_csv, agents_csv)   


    df_clusters = clustering_and_visualization('data/Siren.csv')
    analyze_temporal_trends("data/Lots.csv")

    categorical_variable_analysis(df_lots)
    acp_visualization(df_lots)


files_to_analyze_withc_columns = {
    'data/Lots.csv': ['awardEstimatedPrice', 'awardPrice', 'contractDuration', 'publicityDuration', 'typeOfContract'],
    'data/Agents.csv': ['country','department','city'],
    'data/Criteria.csv': [ 'weight', 'type'],
    'data/LotBuyers.csv': ['agentId'],
    'data/LotSuppliers.csv': ['agentId'],
    
   
    
}
fichiers_a_analyser = [
    'data/Lots.csv',
    'data/Agents.csv',
    'data/Criteria.csv',
    'data/LotBuyers.csv',
    'data/LotSuppliers.csv'
]


processus_analyse_donnees(fichiers_a_analyser,files_to_analyze_withc_columns)


partie_questions()

