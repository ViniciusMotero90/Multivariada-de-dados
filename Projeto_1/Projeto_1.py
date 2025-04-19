import sklearn
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import shapiro, mannwhitneyu
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

df_dsa = pd.read_csv('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_1/dataset.csv')

print(df_dsa.shape)
print(df_dsa.head())
print(df_dsa.info())

df_dsa.columns = df_dsa.columns.str.replace(' ','')

df_dsa['Renda'] = df_dsa['Renda'].str.replace('$','').str.replace(',','').astype(float)
df_dsa['Data_Cadastro_Cliente'] = pd.to_datetime(df_dsa['Data_Cadastro_Cliente'])
msno.matrix(df_dsa)
plt.show()

valores_ausentes = df_dsa.isnull().sum().sort_values(ascending=False)
print(valores_ausentes)
print(valores_ausentes.loc[valores_ausentes != 0])

plt.figure(figsize=(10,6))
sns.boxplot(df_dsa['Renda'])
plt.title('Boxplot da Renda')
plt.xlabel('Renda')
plt.show()

Q1 = df_dsa['Renda'].quantile(0.25)
Q3 = df_dsa['Renda'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_dsa[(df_dsa['Renda'] < lower_bound) | (df_dsa['Renda'] > upper_bound)]
print(outliers)

plt.figure(figsize=(10,6))
sns.distplot(df_dsa['Renda'],color='brown')
plt.title('Distribuição de Renda',size=16)
plt.show()

df_dsa = df_dsa[(df_dsa['Renda'] >= lower_bound) & (df_dsa['Renda'] <= upper_bound)]

plt.figure(figsize=(10,6))
sns.boxplot(df_dsa['Renda'])
plt.title('Boxplot da Renda')
plt.xlabel('Renda')
plt.show()

scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_dsa[['Renda']])

imputer = KNNImputer(n_neighbors=5)
data_imputed = imputer.fit_transform(data_scaled)
df_dsa['Renda'] = scaler.inverse_transform(data_imputed)

print('Total de Valores Ausentes:', df_dsa.isnull().sum().sum())
print(df_dsa.head())

data_boxplot = df_dsa.drop(columns= ['ID',
                                     'Educacao',
                                     'Estado_Civil',
                                     'Criancas_Em_Casa',
                                     'Data_Cadastro_Cliente',
                                     'Adolescentes_Em_Casa',
                                     'Aceitou_Campanha_1',
                                     'Aceitou_Campanha_2',
                                     'Aceitou_Campanha_3',
                                     'Aceitou_Campanha_4',
                                     'Aceitou_Campanha_5',
                                     'Aceitou_Campanha_6'])
print(data_boxplot.columns)

data_boxplot.plot(subplots=True,layout=(4,5),kind='box',figsize=(12,14),patch_artist=True)
plt.subplots_adjust(wspace=0.5)
plt.show()

ano_atual = datetime.now().year
df_dsa['Idade'] = ano_atual - df_dsa['Ano_Nascimento']
df_dsa['Dia_Como_Cliente'] = df_dsa['Data_Cadastro_Cliente'].max() - df_dsa['Data_Cadastro_Cliente']
print(df_dsa.head())
df_dsa['Dia_Como_Cliente'] = df_dsa['Dia_Como_Cliente'].astype(str).str.replace(' days', '')
print(df_dsa.head())
df_dsa['Dia_Como_Cliente'] = pd.to_numeric(df_dsa['Dia_Como_Cliente'], downcast= 'integer')
df_dsa['TotalCompras'] = df_dsa['Num_Compras_Web'] + df_dsa['Num_Compras_Catalogo'] + df_dsa['Num_Compras_Loja']
print(df_dsa.head())
df_dsa['Gasto_Total'] = df_dsa.filter(like='Gasto').sum(axis=1)
print(df_dsa.head())
df_dsa['aceite_campanha'] = df_dsa.filter(like='Aceitou').sum(axis=1)
print(df_dsa.head())
df_dsa['RespostaCampanha'] = df_dsa['aceite_campanha'].apply(lambda x: 'Aceitou' if x > 0 else 'Não Aceitou')
print(df_dsa.head())
print(df_dsa[['Idade','Dia_Como_Cliente','TotalCompras','Gasto_Total','RespostaCampanha']].sample(10))

df_dsa.drop(['Ano_Nascimento','Data_Cadastro_Cliente','aceite_campanha'],axis=1,inplace=True)
print(df_dsa.head())

hist = pd.melt(df_dsa,value_vars=df_dsa)
hist = sns.FacetGrid(hist,col='variable',col_wrap=5,sharex=False,sharey=False)
hist.map(sns.histplot,'value')
plt.show()

fig, ax = plt.subplots(figsize=(6,4))
counts = df_dsa['RespostaCampanha'].value_counts()
labels = counts.index.tolist()

colors = sns.color_palette('husl')

ax.pie(counts,labels=labels,colors=colors,autopct='%.0f%%',startangle=90)
ax.set_title('Qual a Proporção de Clientes que Aceitaram/Não Aceitaram Campanha de Marketing?',fontsize=14)
ax.axis('equal')
plt.show()

Campanhas = ['Aceitou_Campanha_1',
             'Aceitou_Campanha_2',
             'Aceitou_Campanha_3',
             'Aceitou_Campanha_4',
             'Aceitou_Campanha_5',
             'Aceitou_Campanha_6']
campaigns = pd.DataFrame(df_dsa[Campanhas].mean() * 100, columns=['Percent']).reset_index()

plt.figure(figsize=(10,6))
ax = sns.barplot(x='index',y='Percent',data=campaigns.sort_values('Percent',ascending=False))

plt.xlabel('Campanhas',size=15)
plt.ylabel('Percentual %', size=15)
plt.title('Taxa de Sucesso nas Campanhas de Marketing',size=22)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%',
                (p.get_x() + p.get_width() / 2,
                 p.get_height()),
                 ha='center',
                 va='bottom')
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
plt.tight_layout()
plt.show()

df_dsa_corr = df_dsa.drop(['ID',
                           'Educacao',
                           'Estado_Civil',
                           'Criancas_Em_Casa',
                           'Adolescentes_Em_Casa',
                           'Aceitou_Campanha_1',
                           'Aceitou_Campanha_2',
                           'Aceitou_Campanha_3',
                           'Aceitou_Campanha_4',
                           'Aceitou_Campanha_5',
                           'Aceitou_Campanha_6',
                           'Educacao',
                           'RespostaCampanha'],axis=1).corr()
column_corr = df_dsa_corr.loc['Idade']

plt.figure(figsize=(6,18))
sns.heatmap(pd.DataFrame(column_corr.sort_values(ascending=False)),
            annot=True,
            cmap='copper',
            cbar=True,
            square=True,
            fmt='.2f')
plt.title('Matriz de Correlação Para Idade')
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x='RespostaCampanha',y='Gasto_Total',data=df_dsa,palette='bright')
plt.show()

produtos = ['Gasto_Vinhos',
            'Gasto_Frutas',
            'Gasto_Carnes',
            'Gasto_Peixes',
            'Gasto_Doces',
            'Gasto_Outros']

df_produtos = pd.DataFrame(df_dsa[produtos].sum(),columns=['Sum']).reset_index()

plt.figure(figsize=(10,6))
ax = sns.barplot(x= 'index',
                 y= 'Sum',
                 data= df_produtos.sort_values('Sum',ascending=False),
                 palette='viridis')
plt.xlabel('Produtos',size=14)
plt.ylabel('Total',size=14)
for p in ax.patches:
    ax.annotate(p.get_height(),(p.get_x() + p.get_width() / 2, p.get_height()),ha='center',va='bottom')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x='RespostaCampanha',y='Gasto_Vinhos',data=df_dsa,palette='Greens_r')
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x='RespostaCampanha',y='Renda',data=df_dsa,palette='pastel')
plt.show()

sns.lmplot(x='Renda',y='Gasto_Total',data=df_dsa,palette='blue',line_kws={'color':'red'})
plt.show()

df_dsa_imp = pd.get_dummies(df_dsa,columns=['Educacao','Estado_Civil'])
print(df_dsa_imp.columns)

X = df_dsa_imp.drop(['ID',
                    'Gasto_Vinhos',
                    'Gasto_Frutas',
                    'Gasto_Carnes',
                    'Gasto_Peixes',
                    'Gasto_Doces',
                    'Gasto_Outros',
                    'Num_Compras_Web',
                    'Num_Compras_Catalogo',
                    'Num_Compras_Loja',
                    'Aceitou_Campanha_1',
                    'Aceitou_Campanha_2',
                    'Aceitou_Campanha_3',
                    'Aceitou_Campanha_4',
                    'Aceitou_Campanha_5',
                    'Aceitou_Campanha_6',
                    'RespostaCampanha'],axis=1)
y=df_dsa_imp['RespostaCampanha'].map({'Não Aceitou': 0, 'Aceitou': 1})

modelo_rf = RandomForestClassifier(random_state=43)
modelo_dsa = modelo_rf.fit(X,y)
importances = modelo_dsa.feature_importances_
std = np.std([tree.feature_importances_ for tree in modelo_dsa.estimators_], axis = 0)
indices = np.argsort(importances)
plt.figure(1,figsize=(11,10))
plt.title('Importância das Características')
plt.barh(range(X.shape[1]),importances[indices],color='y',xerr=std[indices],align='center')
plt.yticks(range(X.shape[1]),X.columns[indices])
plt.ylim([-1,X.shape[1]])
plt.show()

#Teste Hipóteses
plt.figure(figsize=(10,6))
ax = df_dsa.Educacao.value_counts().plot.bar()
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
ax.bar_label(ax.containers[0])
plt.show()

media_renda = df_dsa.groupby('Educacao')['Renda'].mean().reset_index()
media_renda = media_renda.sort_values(by='Renda')
plt.figure(figsize=(10,6))
ax = sns.barplot(x=media_renda['Educacao'],y=media_renda['Renda'],data=df_dsa,palette='Reds_r')
for p in ax.patches:
    ax.annotate(f'{p.get_height():.0f}',(p.get_x() + p.get_width()/2,p.get_height()),
                ha='center',
                va='bottom')
plt.xlabel('\nNível Educacional')
plt.ylabel('Média de Renda')
plt.title('Renda Média Por Nível Educacional')
plt.show()

df_dsa['Educacao'] = df_dsa['Educacao'].map({'Ensino Fundamental': 1,
                                             'Ensino Médio': 2,
                                             'Graduação': 3,
                                             'Mestrado': 4,
                                             'PhD': 5})
dados_para_testar = df_dsa[['Renda','Educacao']]

def dsa_testa_normal(columns):
    for column in columns:
        statistic, p_value = shapiro(dados_para_testar[column])
        alpha = 0.05
        if p_value < alpha:
            print(f'\n{column}: Alpha {alpha} < valor-p {p_value:.2f} - Rejeitamos a H0 do Teste Shapiro-Wilk: Os dados não são normalmente distribuídos.')
        else:
             print(f'\n{column}: Alpha {alpha} > valor-p {p_value:.2f} - Falhamos em Rejeitar a H0 do Teste Shapiro-Wilk: Os dados seguem uma distribuição normal.')
print(dsa_testa_normal(dados_para_testar))

grupo_com_graduacao = df_dsa[df_dsa['Educacao'].isin([5, 4, 3])]['Renda']
grupo_sem_graduacao = df_dsa[df_dsa['Educacao'].isin([1, 2])]['Renda']

# Teste Mann-Whitney U 
statistic, p_value = mannwhitneyu(grupo_com_graduacao, grupo_sem_graduacao)

# Nível de significância
alpha = 0.05

# Resultado
if p_value < alpha:
    print("Rejeitamos a hipótese nula: Há uma diferença significativa nas médias de renda.")
else:
    print("Falhamos em rejeitar a hipótese nula: Não há diferença significativa nas médias de renda.")