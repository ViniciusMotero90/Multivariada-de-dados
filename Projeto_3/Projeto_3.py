import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn import metrics

dsa_dados = pd.read_csv('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_3/dataset.csv')
print(dsa_dados.shape)
print(dsa_dados.sample(10))
print(dsa_dados.dtypes)

print(dsa_dados.columns)
# Ajusta o dataframe
df_dsa = pd.DataFrame({'country': dsa_dados['Country'],
                       'life_expectancy': dsa_dados['Life expectancy '],
                       'year': dsa_dados['Year'],
                       'status': dsa_dados['Status'],
                       'adult_mortality': dsa_dados['Adult Mortality'],
                       'inf_death': dsa_dados['infant deaths'],
                       'alcohol': dsa_dados['Alcohol'],
                       'hepatitisB': dsa_dados['Hepatitis B'],
                       'measles': dsa_dados['Measles '],
                       'bmi': dsa_dados[' BMI '],
                       'polio': dsa_dados['Polio'],
                       'diphtheria': dsa_dados['Diphtheria '],
                       'hiv': dsa_dados[' HIV/AIDS'],
                       'gdp': dsa_dados['GDP'],
                       'total_expenditure': dsa_dados['Total expenditure'],
                       'thinness_till19': dsa_dados[' thinness  1-19 years'],
                       'thinness_till9': dsa_dados[' thinness 5-9 years'],
                       'school': dsa_dados['Schooling'],
                       'population': dsa_dados[' Population']})

# Cria um dicionário com a descrição de cada variável
dsa_df_dict = {
    "country": "País de origem dos dados.",
    "life_expectancy": "Expectativa de vida ao nascer, em anos.",
    "year": "Ano em que os dados foram coletados.",
    "status": "Status de desenvolvimento do país ('Developing' para países em desenvolvimento, 'Developed' para países desenvolvidos).",
    "adult_mortality": "Taxa de mortalidade de adultos entre 15 e 60 anos por 1000 habitantes.",
    "inf_death": "Número de mortes de crianças com menos de 5 anos por 1000 nascidos vivos.",
    "alcohol": "Consumo de álcool per capita (litros de álcool puro por ano).",
    "hepatitisB": "Cobertura de vacinação contra hepatite B em crianças de 1 ano (%).",
    "measles": "Número de casos de sarampo relatados por 1000 habitantes.",
    "bmi": "Índice médio de massa corporal da população adulta.",
    "polio": "Cobertura de vacinação contra poliomielite em crianças de 1 ano (%).",
    "diphtheria": "Cobertura de vacinação contra difteria, tétano e coqueluche (DTP3) em crianças de 1 ano (%).",
    "hiv": "Prevalência de HIV na população adulta (%).",
    "gdp": "Produto Interno Bruto per capita (em dólares americanos).",
    "total_expenditure": "Gasto total em saúde como porcentagem do PIB.",
    "thinness_till19": "Prevalência de magreza em crianças e adolescentes de 10 a 19 anos (%).",
    "thinness_till9": "Prevalência de magreza em crianças de 5 a 9 anos (%).",
    "school": "Número médio de anos de escolaridade.",
    "population": "População total do país."
}

df_dsa['life_expectancy'].hist()
plt.show()

print(df_dsa.describe(include=['object']))
print(df_dsa.describe(include= 'all'))

def dsa_get_pairs(data,alvo,atributos,n):
    grupos_linhas = [atributos[i:i+n] for i in range(0,len(atributos),n)]
    for linha in grupos_linhas:
        plot = sns.pairplot(x_vars=linha,y_vars=alvo,data=data,kind='reg',height=3)
        plt.show()
    return
alvo = ['life_expectancy']
atributos = ['population','hepatitisB','gdp','total_expenditure','alcohol','school']
dsa_get_pairs(df_dsa,alvo,atributos,3)

print(df_dsa.count())

valores_ausentes = df_dsa.isnull().sum().sort_values(ascending=False)
print(valores_ausentes)

valores_ausentes_percent = valores_ausentes[valores_ausentes > 0] / df_dsa.shape[0]
print(f'{valores_ausentes_percent * 100}%')

atributos = ['population', 'hepatitisB', 'gdp', 'total_expenditure', 'alcohol', 'school']
novo_dataframe = df_dsa[atributos]

Q1 = novo_dataframe.quantile(0.25)
Q3 = novo_dataframe.quantile(0.75)
IQR = Q3 - Q1

outliers = ((novo_dataframe < (Q1 - 1.5 * IQR)) | (novo_dataframe > (Q3 + 1.5 * IQR))).sum()
print(outliers)

outliers_summary = pd.DataFrame({'Outliers': outliers, 'Percentual': (outliers/len(novo_dataframe))*100})
print(outliers_summary)
print(outliers_summary[outliers_summary['Outliers']>0])

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR
novo_df_dsa = df_dsa[~((novo_dataframe < limite_inferior) | (novo_dataframe > limite_superior)).any(axis=1)]
print(novo_df_dsa.shape)

dsa_get_pairs(novo_df_dsa,alvo,atributos,3)

valores_ausentes = novo_df_dsa.isnull().sum().sort_values(ascending=False)
print(valores_ausentes)

valores_ausentes_percent = valores_ausentes[valores_ausentes > 0] / novo_df_dsa.shape[0]
print(f'{valores_ausentes_percent * 100}%')

def impute_median(dados):
    return dados.fillna(dados.median())

novo_df_dsa.loc[:,'population'] = novo_df_dsa['population'].transform(impute_median)
novo_df_dsa.hepatitisB = novo_df_dsa['hepatitisB'].transform(impute_median)
novo_df_dsa.alcohol = novo_df_dsa['alcohol'].transform(impute_median)
novo_df_dsa.total_expenditure = novo_df_dsa['total_expenditure'].transform(impute_median)
novo_df_dsa.gdp = novo_df_dsa['gdp'].transform(impute_median)
novo_df_dsa.school = novo_df_dsa['school'].transform(impute_median)
valores_ausentes = novo_df_dsa.isnull().sum().sort_values(ascending = False)
valores_ausentes_percent = valores_ausentes[valores_ausentes > 0] / novo_df_dsa.shape[0] 
print(f'{valores_ausentes_percent * 100} %')

novo_df_dsa = novo_df_dsa.copy()
novo_df_dsa.dropna(inplace=True)
valores_ausentes = novo_df_dsa.isnull().sum().sort_values(ascending = False)
valores_ausentes_percent = valores_ausentes[valores_ausentes > 0] / novo_df_dsa.shape[0] 
print(f'{valores_ausentes_percent * 100} %')

novo_df_dsa.drop(['country', 'status'], axis = 1, inplace = True)
print(novo_df_dsa.shape)

novo_df_dsa['lifestyle'] = 0
novo_df_dsa.lifestyle = novo_df_dsa['bmi'] * novo_df_dsa['alcohol']

print(novo_df_dsa.lifestyle.describe())

valores_ausentes = novo_df_dsa.isnull().sum().sort_values(ascending=False)
print(valores_ausentes)

print(novo_df_dsa.corr())

def dsa_filtrar_e_visualizar_correlacao(df,threshold,drop_column=None):
    corr = df.corr()
    filtro = (abs(corr) >= threshold) & (corr != 1.0)
    df_filtrando = corr.where(filtro).dropna(how='all').dropna(axis=1,how='all')
    if drop_column:
        df_filtrando = df_filtrando.drop(index=drop_column,errors='ignore').drop(columns=drop_column,errors='ignore')
    plt.figure(figsize=(8,6))
    sns.heatmap(df_filtrando,annot=True,cmap='coolwarm',center=0)
    plt.show()
dsa_filtrar_e_visualizar_correlacao(novo_df_dsa,threshold=0.3,drop_column=None)
dsa_filtrar_e_visualizar_correlacao(novo_df_dsa,threshold=0.55,drop_column='life_expectancy')

novo_df_final = pd.DataFrame({'life_expectancy': novo_df_dsa['life_expectancy'],
                              'adult_mortality': novo_df_dsa['adult_mortality'],
                              'diphtheria': novo_df_dsa['diphtheria'],
                              'hiv': novo_df_dsa['hiv'],
                              'gdp': novo_df_dsa['gdp'],
                              'thinness_till19': novo_df_dsa['thinness_till19'],
                              'school': novo_df_dsa['school'],
                              'lifestyle': novo_df_dsa['lifestyle'],})

missing_values = novo_df_final.isnull().sum().sort_values(ascending=False)
print(missing_values)

X = novo_df_final.drop('life_expectancy',axis=1)
y = novo_df_final['life_expectancy']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

modelo = RandomForestRegressor(n_estimators=100,random_state=42)
modelo.fit(X_train,y_train)

y_pred = modelo.predict(X_test)

mse = metrics.mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
print(f'RMSE: {rmse}')
print(f'R2 score: {metrics.r2_score(y_test,y_pred)}')

importancias = modelo.feature_importances_
variaveis = X.columns

importancias_df = pd.DataFrame({'Variável': variaveis,
                                'Importância':importancias}).sort_values(by='Importância',ascending=False)
print(importancias_df)

novo_df_final = novo_df_final.drop('gdp',axis=1)
print(novo_df_final.head())

X = novo_df_final[['hiv', 'adult_mortality', 'school', 'thinness_till19', 'lifestyle', 'diphtheria']].values
y = novo_df_final.life_expectancy.values.reshape(-1,1)

x_treino,x_teste,y_treino,y_teste = train_test_split(X,y,test_size=0.2,random_state=0)
dsa_scaler = StandardScaler()
dsa_scaler.fit(x_treino)
x_treino_scaled = dsa_scaler.transform(x_treino)
x_teste_scaled = dsa_scaler.transform(x_teste)

print(x_treino_scaled)
print(x_treino_scaled.shape)
print(x_teste_scaled.shape)

modelo_dsa_v1 = LinearRegression()

modelo_dsa_v1.fit(x_treino_scaled,y_treino)

print(f'Coeficiente: \n {modelo_dsa_v1.coef_}')

y_pred_treino_v1 = modelo_dsa_v1.predict(x_treino_scaled)

print(f'Mean Absolute Error: {metrics.mean_absolute_error(y_treino,y_pred_treino_v1)}')
print(f'Mean Squared Error: {metrics.mean_squared_error(y_treino,y_pred_treino_v1)}')
print(f'Root Mean Squared Error: {metrics.root_mean_squared_error(y_treino,y_pred_treino_v1)}')
print(f'R2 Score: {metrics.r2_score(y_treino,y_pred_treino_v1)}')

y_pred_teste_v1 = modelo_dsa_v1.predict(x_teste_scaled)

df_previsoes = pd.DataFrame({'Valor_Real': y_teste.flatten(),'Valor_Previsto': y_pred_teste_v1.flatten()})
print(df_previsoes.head())

def dsa_cria_scatter(x,y,title,xlabel,ylabel):
    fig,ax = plt.subplots(figsize=(10,6))
    ax.scatter(x,y,color='blue',alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
    return

dsa_cria_scatter(df_previsoes.Valor_Real,df_previsoes.Valor_Previsto,'Modelo','Previsões','Reais')

print(f'Mean Absolute Error: {metrics.mean_absolute_error(y_teste,y_pred_teste_v1)}')
print(f'Mean Squared Error: {metrics.mean_squared_error(y_teste,y_pred_teste_v1)}')
print(f'Root Mean Squared Error: {metrics.root_mean_squared_error(y_teste,y_pred_teste_v1)}')
print(f'R2 Score: {metrics.r2_score(y_teste,y_pred_teste_v1)}')

modelo_dsa_v2 = Lasso(alpha=1.0)
print(modelo_dsa_v2.fit(x_treino_scaled,y_treino))
y_pred_treino_v2 = modelo_dsa_v2.predict(x_treino_scaled)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_treino, y_pred_treino_v2))
print('Mean Squared Error:', metrics.mean_squared_error(y_treino, y_pred_treino_v2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_treino, y_pred_treino_v2)))
print('R2 Score:', metrics.r2_score(y_treino, y_pred_treino_v2))

y_pred_teste_v2 = modelo_dsa_v2.predict(x_teste_scaled)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_teste, y_pred_teste_v2))
print('Mean Squared Error:', metrics.mean_squared_error(y_teste, y_pred_teste_v2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_teste, y_pred_teste_v2)))
print('R2 Score:', metrics.r2_score(y_teste, y_pred_teste_v2))

modelo_dsa_v3 = Ridge(alpha=1.0)
modelo_dsa_v3.fit(x_treino_scaled,y_treino)
y_pred_treino_v3 = modelo_dsa_v3.predict(x_treino_scaled)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_treino, y_pred_treino_v3))
print('Mean Squared Error:', metrics.mean_squared_error(y_treino, y_pred_treino_v3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_treino, y_pred_treino_v3)))
print('R2 Score:', metrics.r2_score(y_treino, y_pred_treino_v3))

y_pred_teste_v3 = modelo_dsa_v3.predict(x_teste_scaled)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_teste, y_pred_teste_v3))
print('Mean Squared Error:', metrics.mean_squared_error(y_teste, y_pred_teste_v3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_teste, y_pred_teste_v3)))
print('R2 Score:', metrics.r2_score(y_teste, y_pred_teste_v3))

modelo_dsa_v4 = ElasticNet(alpha = 1.0, l1_ratio = 0.5)
modelo_dsa_v4.fit(x_treino_scaled, y_treino)
y_pred_treino_v4 = modelo_dsa_v4.predict(x_treino_scaled)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_treino, y_pred_treino_v4))
print('Mean Squared Error:', metrics.mean_squared_error(y_treino, y_pred_treino_v4))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_treino, y_pred_treino_v4)))
print('R2 Score:', metrics.r2_score(y_treino, y_pred_treino_v4))

y_pred_teste_v4 = modelo_dsa_v4.predict(x_teste_scaled)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_teste, y_pred_teste_v4))
print('Mean Squared Error:', metrics.mean_squared_error(y_teste, y_pred_teste_v4))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_teste, y_pred_teste_v4)))
print('R2 Score:', metrics.r2_score(y_teste, y_pred_teste_v4))

modelo_dsa_v5 = Ridge()
parametros = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid_search_dsa = GridSearchCV(estimator = modelo_dsa_v5, 
                               param_grid = parametros, 
                               cv = 5, 
                               scoring = 'neg_mean_squared_error', 
                               verbose = 1)
grid_search_dsa.fit(x_treino_scaled, y_treino)
melhor_modelo = grid_search_dsa.best_estimator_
y_pred_treino_v5 = melhor_modelo.predict(x_treino_scaled)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_treino, y_pred_treino_v5))
print('Mean Squared Error:', metrics.mean_squared_error(y_treino, y_pred_treino_v5))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_treino, y_pred_treino_v5)))
print('R2 Score:', metrics.r2_score(y_treino, y_pred_treino_v5))

y_pred_teste_v5 = melhor_modelo.predict(x_teste_scaled)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_teste, y_pred_teste_v5))
print('Mean Squared Error:', metrics.mean_squared_error(y_teste, y_pred_teste_v5))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_teste, y_pred_teste_v5)))
print('R2 Score:', metrics.r2_score(y_teste, y_pred_teste_v5))
print('Melhor alpha:', grid_search_dsa.best_params_['alpha'])

print('RMSE V1:', np.sqrt(metrics.mean_squared_error(y_teste, y_pred_teste_v1)))
print('RMSE V2:', np.sqrt(metrics.mean_squared_error(y_teste, y_pred_teste_v2)))
print('RMSE V3:', np.sqrt(metrics.mean_squared_error(y_teste, y_pred_teste_v3)))
print('RMSE V4:', np.sqrt(metrics.mean_squared_error(y_teste, y_pred_teste_v4)))
print('RMSE V5:', np.sqrt(metrics.mean_squared_error(y_teste, y_pred_teste_v5)))

print('R2 Score Modelo V1:', metrics.r2_score(y_teste, y_pred_teste_v1))
print('R2 Score Modelo V2:', metrics.r2_score(y_teste, y_pred_teste_v2))
print('R2 Score Modelo V3:', metrics.r2_score(y_teste, y_pred_teste_v3))
print('R2 Score Modelo V4:', metrics.r2_score(y_teste, y_pred_teste_v4))
print('R2 Score Modelo V5:', metrics.r2_score(y_teste, y_pred_teste_v5))

# Calculando os resíduos para o conjunto de treino
residuos_treino = y_treino - y_pred_treino_v1

# Calculando os resíduos para o conjunto de teste
residuos_teste = y_teste - y_pred_teste_v1

# Plotando os resíduos do conjunto de treino
plt.figure(figsize = (10, 5))
plt.scatter(y_pred_treino_v1, residuos_treino, color = 'blue', label = 'Treino', alpha = 0.5)
plt.axhline(y = 0, color = 'red', linestyle = '--')
plt.xlabel('Valores Previstos')
plt.ylabel('Resíduos')
plt.title('Resíduos vs. Valores Previstos (Treino)')
plt.legend()
plt.show()

# Plotando os resíduos do conjunto de teste
plt.figure(figsize = (10, 5))
plt.scatter(y_pred_teste_v1, residuos_teste, color = 'green', label = 'Teste', alpha = 0.5)
plt.axhline(y = 0, color = 'red', linestyle = '--')
plt.xlabel('Valores Previstos')
plt.ylabel('Resíduos')
plt.title('Resíduos vs. Valores Previstos (Teste)')
plt.legend()
plt.show()
joblib.dump(dsa_scaler, 'C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_3/dsa_scaler.pkl')
joblib.dump(modelo_dsa_v1, 'C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_3/modelo_dsa_v1.pkl')

scaler_final = joblib.load('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_3/dsa_scaler.pkl')
modelo_final = joblib.load('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_3/modelo_dsa_v1.pkl')

novos_dados = pd.read_csv('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_3/novos_dados.csv')

novos_dados_scaled = scaler_final.transform(novos_dados)
previsao = modelo_final.predict(novos_dados_scaled)
print('De acordo com os dados de entrada a expectativa de vida (em anos) é de aproximadamente:', 
      np.round(previsao, 2))