import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score,LeaveOneOut
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
dados = np.random.randn(1000,8)
df_dsa = pd.DataFrame(dados,columns=[f'Atributo_{i}' for i in range(1,9)])
print(df_dsa)

dsa_scaler = StandardScaler()

dados_padronizados = dsa_scaler.fit_transform(df_dsa)

dsa_PCA = PCA()

dados_pca = dsa_PCA.fit_transform(dados_padronizados)

cov_matrix = np.cov(dados_padronizados,rowvar=False)
print(cov_matrix)

explained_variance = dsa_PCA.explained_variance_ratio_
componentes = range(1,len(explained_variance)+1)

plt.bar(componentes,explained_variance)
plt.xticks(componentes)
plt.xlabel('Componente Principal')
plt.ylabel('Variância Explicada')
plt.title('Variância Explicada por Cada Componente')
plt.show()

componentes = pd.DataFrame(dsa_PCA.components_,columns=df_dsa.columns,index=[f'PC-{i}' for i in range(1,9)])

print(componentes)

df_dsa['target'] = np.random.randint(0,2,df_dsa.shape[0])
print(df_dsa.head())

scores = []

for i in range(1,9):
    pca = PCA(n_components=i)
    pca_data = pca.fit_transform(dados_padronizados)
    lr = LogisticRegression()
    score = cross_val_score(lr,pca_data,df_dsa['target'],cv=5).mean()
    scores.append(score)

plt.plot(range(1,9),scores)
plt.xlabel('Número de Componentes')
plt.ylabel('Desempenho Médio em Validação Cruzada')
plt.title('Desempenho vs. Número de Componentes')
plt.show()

dsa_loo = LeaveOneOut()
dsa_lr = LogisticRegression()
scores_loo = cross_val_score(dsa_lr,pca_data,df_dsa['target'],cv=dsa_loo)
print(f'LOOCV Score: {scores_loo.mean()}')

pca_final = PCA(n_components=6)
pca_data_final = pca_final.fit_transform(dados_padronizados)
modelo_dsa_lr_final = LogisticRegression()
modelo_dsa_lr_final.fit(pca_data_final,df_dsa['target'])

novo_dados_sensores = np.random.randn(1,8)
novo_dados_padronizados = dsa_scaler.transform(novo_dados_sensores)
pca_new_data = pca_final.transform(novo_dados_padronizados)

previsao = modelo_dsa_lr_final.predict(pca_new_data)
print(previsao)