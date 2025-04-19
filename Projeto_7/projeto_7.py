import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from sklearn import metrics

df_dsa = pd.read_csv('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_7/transaction_dataset.csv')
print(df_dsa.head())
print(df_dsa.FLAG.value_counts())

df_dsa.columns = [x.lower() for x in df_dsa.columns]

cols_to_drop = [' erc20 most sent token type',
                ' erc20_most_rec_token_type',
                'address',
                'index',
                'unnamed: 0']
atributos = [x for x in df_dsa.columns if(x != 'flag' and x not in cols_to_drop)]
print(atributos)

valores_unicos = df_dsa.nunique()
print(valores_unicos)

atributos = [x for x in atributos if x in valores_unicos.loc[(valores_unicos>1)]]
print(df_dsa[atributos].info())

class DSAPipeSteps(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[]):
        self.columns = columns
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        return X
    
class DSASelecionaColunas(DSAPipeSteps):
    def transform(self, X):
        X = X.copy()
        return X[self.columns]

class DSAPreencheDados(DSAPipeSteps):

    def fit(self, X, y=None):
        self.means = { col: X[col].mean() for col in self.columns }
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].fillna(self.means[col])
        return X

# Definição de uma classe que herda de DSAPipeSteps
class DSAPadronizaDados(DSAPipeSteps):

    # Método fit para ajustar o scaler nos dados de treinamento
    def fit(self, X, y = None):
        
        # Inicializa uma instância de StandardScaler para padronizar os dados
        self.scaler = StandardScaler()
        
        # Ajusta o scaler nas colunas especificadas em self.columns
        self.scaler.fit(X[self.columns])
        
        # Retorna a própria instância, indicando que o método não faz modificações
        return self

    # Método transform para transformar os dados de entrada
    def transform(self, X):
        
        # Faz uma cópia dos dados de entrada para evitar alterar os dados originais
        X = X.copy()
        
        # Aplica a transformação de padronização nas colunas especificadas
        X[self.columns] = self.scaler.transform(X[self.columns])
        
        # Retorna os dados transformados
        return X

dsa_pipe_preprocessamento = Pipeline([('feature_selection',DSASelecionaColunas(atributos)),
                                      ('fill_missing',DSAPreencheDados(atributos)),
                                      ('standard_scaling',DSAPadronizaDados(atributos))
                                    ])
dsa_pipe_final = Pipeline([('preprocessing',dsa_pipe_preprocessamento),
                           ('learning',XGBClassifier(random_state = 42, eval_metric = 'auc', objective = 'binary:logistic'))
                          ])
X = df_dsa[atributos]
y = df_dsa['flag']

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.30, random_state = 42)

dsa_pipe_final.fit(X_treino,y_treino)
previsoes_teste = dsa_pipe_final.predict(X_teste)
score_auc = metrics.roc_auc_score(y_teste,previsoes_teste)
print(f'AUC nos Dados de Teste - {score_auc:,.2%}')
dump(dsa_pipe_final,'C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_7/modelo_dsa_final.joblib')

novos_dados = pd.read_csv('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_7/novos_dados.csv')

modelo_carregado = load('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_7/modelo_dsa_final.joblib')
previsao = modelo_carregado.predict(novos_dados)
print(previsao)

if previsao[0] == 0:
    print("Segundo o modelo, provavelmente, essa transação não representa uma Fraude.")
else:
    print("Segundo o modelo, provavelmente, essa transação pode representar uma Fraude. Acione verificação humana!")