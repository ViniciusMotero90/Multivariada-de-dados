import pandas as pd
import numpy as np
import pickle

df_dsa = pd.read_csv('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_2/dados/dataset.csv')
print(df_dsa.shape)
print(df_dsa.head())
print(df_dsa.LABEL_TARGET.value_counts())
df_dsa['LABEL_TARGET'] = df_dsa['LABEL_TARGET'].astype(int)
print(df_dsa.head())
print(df_dsa.isnull().values.any())
lista_de_colunas = df_dsa.columns.to_list()
colunas_entrada = lista_de_colunas[0:178]
print(colunas_entrada)

dup_cols = set([x for x in colunas_entrada if colunas_entrada.count(x) > 1])
print(dup_cols)

dup_cols = set([x for x in lista_de_colunas if lista_de_colunas.count(x) > 1])
print(dup_cols)

def calcula_prevalencia(y_actual):
    return sum(y_actual) / len(y_actual)
print(f"Prevalência da classe positiva: {calcula_prevalencia(df_dsa['LABEL_TARGET'].values):.3f}")

df_data = df_dsa.sample(n=len(df_dsa))
df_data = df_data.reset_index(drop=True)
df_amostra_30 = df_data.sample(frac=0.3)
print(f"Tamanho da divisão de validação / teste: {(len(df_amostra_30)/len(df_data)):.1f}")

df_teste = df_amostra_30.sample(frac=0.5)
df_valid = df_amostra_30.drop(df_teste.index)
df_treino = df_data.drop(df_amostra_30.index)

print(f"Teste(n= {len(df_teste)}): {calcula_prevalencia(df_teste.LABEL_TARGET.values):.3f}")
print(f"Validação(n= {len(df_valid)}): {calcula_prevalencia(df_valid.LABEL_TARGET.values):.3f}")
print(f"Treino(n= {len(df_treino)}): {calcula_prevalencia(df_treino.LABEL_TARGET.values):.3f}")

print(df_treino.LABEL_TARGET.value_counts())

indice = df_treino.LABEL_TARGET == 1

df_train_pos = df_treino[indice]
df_train_neg = df_treino[~indice]

valor_minimo = np.min([len(df_train_pos),len(df_train_neg)])
print(valor_minimo)

df_treino_final = pd.concat([df_train_pos.sample(n=valor_minimo,random_state=69),
                             df_train_neg.sample(n=valor_minimo,random_state=69)],
                             axis=0,
                             ignore_index=True)
df_treino_final = df_treino_final.sample(n=len(df_treino_final),random_state=69).reset_index(drop=True)
print(df_treino_final.shape)
print(df_treino_final.LABEL_TARGET.value_counts())

print(f'Balanceamento em Treino(n={len(df_treino_final)}): {calcula_prevalencia(df_treino_final.LABEL_TARGET.values)}')

df_treino.to_csv('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_2/dados/dados_treino.csv',index=False)
df_treino_final.to_csv('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_2/dados/dados_treino_final.csv',index=False)
df_valid.to_csv('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_2/dados/dados_valid.csv',index=False)
df_teste.to_csv('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_2/dados/dados_teste.csv',index=False)
pickle.dump(colunas_entrada,open('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_2/dados/colunas_entrada.sav','wb'))