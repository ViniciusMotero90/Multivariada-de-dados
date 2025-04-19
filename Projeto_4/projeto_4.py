import threadpoolctl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

np.random.seed(42)
df_dsa = pd.read_csv('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_4/dataset.csv')

print(df_dsa.shape)
print(df_dsa.info())
print(df_dsa.head())
print(df_dsa.sample(10))
print(df_dsa.isnull().sum())

for column in df_dsa.columns:
    if df_dsa[column].dtype in ['int64','float64']:
        plt.figure(figsize=(5,5))
        sns.boxplot(x=df_dsa[column])
        plt.title(column)
        plt.show()

print(df_dsa.describe())
Q1 = df_dsa.quantile(0.25)
Q3 = df_dsa.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

outliers = ((df_dsa < (Q1 - 2.5 * IQR)) | (df_dsa > (Q3 + 2.5 * IQR))).any(axis=1)

df_outliers = df_dsa[outliers]
print(df_outliers)

correlation_matrix = df_dsa.corr()

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_dsa),columns=df_dsa.columns)
sns.pairplot(df_dsa,hue='Cancelou',diag_kind='kde')
plt.show()

#Método do Cotovelo
sse = []
k_range = range(1,11)
for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df_scaled)
    sse.append(kmeans.inertia_)
plt.plot(k_range,sse,'bx-')
plt.xlabel('k')
plt.ylabel('Soma dos Quadrados Intra-Cluster')
plt.title('Método do Cotovelo para Ótimo k')
plt.show()

#Método da Silhueta
sil_score = []
for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df_scaled)
    if k != 1:
        sil_score.append(silhouette_score(df_scaled,kmeans.labels_))
plt.plot(k_range[1:],sil_score,'bx-')
plt.xlabel('k')
plt.ylabel('Coeficiente de Silhueta')
plt.title('Método do Silhueta para Ótimo k')
plt.show()

modelo_kmeans_dsa = KMeans(n_clusters=4,random_state=42)
df_dsa['cluster'] = modelo_kmeans_dsa.fit_predict(df_scaled)
print(df_dsa.sample(10))
df_dsa_cleaned = df_dsa

# Criando um mapa de cores baseado na paleta 'Dark2'
palette = sns.color_palette('Dark2', n_colors = len(df_dsa_cleaned['cluster'].unique()))
color_map = dict(zip(df_dsa_cleaned['cluster'].unique(), palette))

# Plotando o gráfico de grid com os clusters e mostrando o mapa de cores
g = sns.PairGrid(df_dsa_cleaned, hue = 'cluster', palette = color_map, diag_sharey = False)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw = 2)
plt.show()

# Mostrando o mapa de cores
for cluster, color in color_map.items():
    plt.scatter([], [], c = [color], label = f'Cluster {cluster}')
plt.legend(title = 'Legenda de Cores dos Clusters')
plt.axis('off')
plt.show()

modelo_kmeans_dsa = KMeans(n_clusters=3,random_state=42)
color_map = dict(zip(df_dsa_cleaned['cluster'].unique(), palette))

df_dsa.drop('cluster', axis = 1, inplace = True)
print(df_dsa.head())
df_dsa['cluster'] = modelo_kmeans_dsa.fit_predict(df_scaled)
print(df_dsa.sample(10))
df_dsa_cleaned = df_dsa[~outliers]
g = sns.PairGrid(df_dsa_cleaned, hue = 'cluster', palette = color_map, diag_sharey = False)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw = 2)
plt.show()

# Mostrando o mapa de cores
for cluster, color in color_map.items():
    plt.scatter([], [], c = [color], label = f'Cluster {cluster}')
plt.legend(title = 'Legenda de Cores dos Clusters')
plt.axis('off')
plt.show()

print(df_dsa_cleaned.head(5))
print(df_dsa_cleaned[df_dsa_cleaned.cluster == 0].head())
print(df_dsa_cleaned[df_dsa_cleaned.cluster == 0].mean())
print(df_dsa_cleaned[df_dsa_cleaned.cluster == 1].head())
print(df_dsa_cleaned[df_dsa_cleaned.cluster == 1].mean())
print(df_dsa_cleaned[df_dsa_cleaned.cluster == 2].head())
print(df_dsa_cleaned[df_dsa_cleaned.cluster == 2].mean())

centroides = modelo_kmeans_dsa.cluster_centers_
print(centroides)

# Cria a figura
plt.figure(figsize = (8, 6))

# Loop pelos clusters
for cluster_num in range(3):
    mask = df_dsa['cluster'] == cluster_num
    plt.scatter(df_scaled[mask].iloc[:, 0], df_scaled[mask].iloc[:, 1], label = f'Cluster {cluster_num}')

# Plot
plt.scatter(modelo_kmeans_dsa.cluster_centers_[:, 0], 
            modelo_kmeans_dsa.cluster_centers_[:, 1], 
            s = 300, 
            c = 'red', 
            marker = 'X', 
            label = 'Centróides')
plt.legend()
plt.title("Cluster Plot")
plt.show()
print(df_dsa['cluster'].value_counts())
print(df_dsa.groupby('cluster').mean())
score = silhouette_score(df_scaled,df_dsa['cluster'])
print(score)

# Plot
plt.scatter(df_dsa[df_dsa['cluster'] == 0]['Idade'], df_dsa[df_dsa['cluster'] == 0]['Gasto_Mensal'], label='Cluster 0')
plt.scatter(df_dsa[df_dsa['cluster'] == 1]['Idade'], df_dsa[df_dsa['cluster'] == 1]['Gasto_Mensal'], label='Cluster 1')
plt.scatter(df_dsa[df_dsa['cluster'] == 2]['Idade'], df_dsa[df_dsa['cluster'] == 2]['Gasto_Mensal'], label='Cluster 2')
plt.legend()
plt.show()

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(df_dsa[df_dsa['cluster'] == 0]['Idade'], df_dsa[df_dsa['cluster'] == 0]['Gasto_Mensal'], df_dsa[df_dsa['cluster'] == 0]['Tempo_de_Assinatura'], label='Cluster 0')
ax.scatter(df_dsa[df_dsa['cluster'] == 1]['Idade'], df_dsa[df_dsa['cluster'] == 1]['Gasto_Mensal'], df_dsa[df_dsa['cluster'] == 1]['Tempo_de_Assinatura'], label='Cluster 1')
ax.scatter(df_dsa[df_dsa['cluster'] == 2]['Idade'], df_dsa[df_dsa['cluster'] == 2]['Gasto_Mensal'], df_dsa[df_dsa['cluster'] == 2]['Tempo_de_Assinatura'], label='Cluster 2')

ax.legend()
plt.show()

# Cria o modelo PCA com 2 componentes principais
pca = PCA(n_components = 2)

# Treina o modelo usando dados padronizados
principalComponents = pca.fit_transform(df_scaled)

# Cria o dataframe com o resultado
df_principal = pd.DataFrame(data = principalComponents, columns = ['PC 1', 'PC 2'])
df_principal['cluster'] = df_dsa['cluster']

# Cria o gráfico
plt.scatter(df_principal[df_principal['cluster'] == 0]['PC 1'], df_principal[df_principal['cluster'] == 0]['PC 2'], label='Cluster 0')
plt.scatter(df_principal[df_principal['cluster'] == 1]['PC 1'], df_principal[df_principal['cluster'] == 1]['PC 2'], label='Cluster 1')
plt.scatter(df_principal[df_principal['cluster'] == 2]['PC 1'], df_principal[df_principal['cluster'] == 2]['PC 2'], label='Cluster 2')
plt.legend()
plt.show()

tsne = TSNE(n_components = 2)
tsne_results = tsne.fit_transform(df_scaled)
df_tsne = pd.DataFrame(data = tsne_results, columns = ['tsne 1', 'tsne 2'])
df_tsne['cluster'] = df_dsa['cluster']

plt.scatter(df_tsne[df_tsne['cluster'] == 0]['tsne 1'], df_tsne[df_tsne['cluster'] == 0]['tsne 2'], label='Cluster 0')
plt.scatter(df_tsne[df_tsne['cluster'] == 1]['tsne 1'], df_tsne[df_tsne['cluster'] == 1]['tsne 2'], label='Cluster 1')
plt.scatter(df_tsne[df_tsne['cluster'] == 2]['tsne 1'], df_tsne[df_tsne['cluster'] == 2]['tsne 2'], label='Cluster 2')
plt.legend()
plt.show()

# Cria a figura
plt.figure(figsize = (8, 6))

# Loop pelos clusters
for cluster_num in range(3):
    mask = df_dsa['cluster'] == cluster_num
    plt.scatter(df_scaled[mask].iloc[:, 0], df_scaled[mask].iloc[:, 1], label = f'Cluster {cluster_num}')

# Plot
plt.scatter(modelo_kmeans_dsa.cluster_centers_[:, 0], 
            modelo_kmeans_dsa.cluster_centers_[:, 1], 
            s = 300, 
            c = 'red', 
            marker = 'X', 
            label = 'Centróides')
plt.legend()
plt.title("Cluster Plot")
plt.show()

pca = PCA(n_components = 2)

pca_result = pca.fit_transform(df_scaled)

df_dsa['pca_1'] = pca_result[:, 0]
df_dsa['pca_2'] = pca_result[:, 1]

plt.figure(figsize=(8, 6))

for cluster_num in range(3):
    mask = df_dsa['cluster'] == cluster_num
    plt.scatter(df_dsa[mask]['pca_1'], df_dsa[mask]['pca_2'], label = f'Cluster {cluster_num}')

# Obtenha os centroides transformados para a visualização
centroids_pca = pca.transform(modelo_kmeans_dsa.cluster_centers_)
plt.scatter(centroids_pca[:, 0], 
            centroids_pca[:, 1], 
            s = 300, 
            c = 'red', 
            marker = 'X', 
            label = 'Centróides')
plt.legend()
plt.title("Cluster Plot com PCA")
plt.show()