import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.manifold import TSNE
from pandas.plotting import parallel_coordinates
import warnings


df_dsa1 = pd.read_csv("C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Capitulo_5/dataset1.csv")
print(df_dsa1.shape)
print(df_dsa1.head())

sns.lmplot(x="Value",y="Overall",hue="Position",data=df_dsa1.loc[df_dsa1['Position'].isin(['ST','RW','LW'])],fit_reg=False)
plt.show()

sns.lmplot(x="Value",y="Overall",markers=['D','x','*'],hue="Position",data=df_dsa1.loc[df_dsa1['Position'].isin(['ST','RW','LW'])],fit_reg=False)
plt.show()

cols_to_convert = ['Overall', 'Potential', 'Acceleration', 'Dribbling', 'Finishing', 'Strength']
df_dsa1[cols_to_convert] = df_dsa1[cols_to_convert].apply(pd.to_numeric,errors='coerce')

plt.figure(figsize=(10,6))
sns.scatterplot(data=df_dsa1,x='Age',y='Overall',hue='Potential',palette='viridis',alpha=0.6)
plt.title('Idade x Avaliação Geral Por Potencial')
plt.xlabel('Idade')
plt.ylabel('Avaliação Geral')
plt.grid(True)
plt.show()

df = (df_dsa1.loc[df_dsa1['Position'].isin(['ST','GK'])].loc[:,['Value','Overall','Aggression','Position']])
df = df[df['Overall']>=80]
df = df[df['Overall']<85]
df['Aggression'] = df['Aggression'].astype(float)
sns.boxplot(x='Overall',y='Aggression',hue='Position',data=df)
plt.show()

df_dsa2 = pd.read_csv('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Capitulo_5/dataset2.csv')

sns.heatmap(df_dsa2.loc[:,['HP','Attack','Sp. Atk','Defense','Sp. Def','Speed']].corr(),annot=True)
plt.show()

top_types = df_dsa2['Type 1'].value_counts()[:10]
df_filtro = df_dsa2[df_dsa2['Type 1'].isin(top_types.index)]

plt.figure(figsize=(12,6))
sns.swarmplot(x='Type 1', y='Total', data=df_filtro, hue='Legendary')
plt.axhline(df_filtro['Total'].mean(),color='red',linestyle='dashed')
plt.show()

dados = (df_dsa2[(df_dsa2['Type 1'].isin(["Psychic","Fighting"]))].loc[:,['Type 1','Attack','Sp. Atk','Defense','Sp. Def']])

plt.figure(figsize=(12,6))
parallel_coordinates(dados,'Type 1',colormap='winter')
plt.show()

df_dsa3 = pd.read_csv('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Capitulo_5/dataset3.csv')
sns.pairplot(df_dsa3,hue='species')
plt.suptitle("Pair Plot",y=1.02)
plt.show()

labels = np.array(['sepal length','sepal width','petal length','petal width'])
stats = df_dsa3.drop('species',axis=1).mean().values
angles = np.linspace(0,2*np.pi,len(labels),endpoint=False).tolist()
stats = np.concatenate((stats,[stats[0]]))
angles += angles[:1]
fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
ax.fill(angles, stats, color='red', alpha=0.25)
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
plt.title("Radar Chart")
plt.show()

plt.figure(figsize=(8,6))
sns.violinplot(data=df_dsa3.drop('species',axis=1))
plt.title('Violin Plot')
plt.show()

tsne_dsa = TSNE(n_components=2,random_state=0)
modelo_tsne = tsne_dsa.fit_transform(df_dsa3.drop('species',axis=1))
plt.figure(figsize=(8,6))
sns.scatterplot(x=modelo_tsne[:,0],y=modelo_tsne[:,1],hue=df_dsa3['species'])
plt.title("TSNE for Multidimensional Data Visualization")
plt.show()