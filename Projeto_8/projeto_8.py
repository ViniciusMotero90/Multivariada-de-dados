import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
num_samples = 1000

taxa_juros = np.random.uniform(0,15,num_samples)
taxa_cambio = np.random.uniform(0,15,num_samples)
producao_industrial = np.random.uniform(50,200,num_samples)

pib = 2 * taxa_juros + 3 * taxa_cambio + 0.5 * producao_industrial + np.random.normal(0,5,num_samples)
inflacao = 0.5 * taxa_juros + 2 * taxa_cambio + 0.2 * producao_industrial + np.random.normal(0,2,num_samples)
taxa_desemprego = -0.1 * taxa_juros + 0.3 * taxa_cambio + 0.4 * producao_industrial + np.random.normal(0,1,num_samples)

df_dsa = pd.DataFrame({'taxa_juros': taxa_juros,
                       'taxa_cambio': taxa_cambio,
                       'producao_industrial': producao_industrial,
                       'pib': pib,
                       'inflacao': inflacao,
                       'taxa_desemprego': taxa_desemprego})

print(df_dsa.head())

X = df_dsa[['taxa_juros','taxa_cambio','producao_industrial']]
y = df_dsa[['pib','inflacao','taxa_desemprego']]

X_treino, X_teste, y_treino, y_teste = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

modelo_dsa = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,random_state=42))

modelo_dsa.fit(X_treino_scaled,y_treino)

y_pred = modelo_dsa.predict(X_teste_scaled)
mse = mean_squared_error(y_teste,y_pred,multioutput='raw_values')
print('Mean Squared Error de cada variável alvo:',mse)

r2 = r2_score(y_teste,y_pred,multioutput='raw_values')
print('R2 de cada variável alvo:',r2)

resultados = pd.DataFrame(y_teste,columns=['pib','inflacao','taxa_desemprego'])
resultados['pib_pred'] = y_pred[:,0]
resultados['inflacao_pred'] = y_pred[:,1]
resultados['taxa_desemprego_pred'] = y_pred[:,2]
print(resultados.head())