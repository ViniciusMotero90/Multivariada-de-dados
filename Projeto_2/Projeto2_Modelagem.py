import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import GridSearchCV

df_treino = pd.read_csv('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_2/dados/dados_treino.csv')
df_treino_final = pd.read_csv('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_2/dados/dados_treino_final.csv')
df_valid = pd.read_csv('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_2/dados/dados_valid.csv')
df_teste = pd.read_csv('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_2/dados/dados_teste.csv')

with open('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_2/dados/colunas_entrada.sav','rb') as file:
    colunas_entrada = pickle.load(file)

X_treino = df_treino_final[colunas_entrada].values
X_valid = df_valid[colunas_entrada].values
X_teste = df_teste[colunas_entrada].values

y_treino = df_treino_final['LABEL_TARGET'].values
y_valid = df_valid['LABEL_TARGET'].values
y_teste = df_teste['LABEL_TARGET'].values

print('Shape dos dados de treino:',X_treino.shape,y_treino.shape)
print('Shape dos dados de validação:',X_valid.shape,y_valid.shape)
print('Shape dos dados de teste:',X_teste.shape,y_teste.shape)

scaler = StandardScaler()
scaler.fit(X_treino)
scalerfile = 'C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_2/dados/scaler.sav'
pickle.dump(scaler,open(scalerfile,'wb'))

scaler = pickle.load(open(scalerfile,'rb'))

X_treino_tf = scaler.transform(X_treino)
X_valid_tf = scaler.transform(X_valid)
print(X_treino_tf)

def dsa_calcula_especificidade(y_actual,y_pred,thresh):
    return sum((y_pred < thresh) & (y_actual == 0)) / sum(y_actual == 0)

def dsa_print_report(y_actual,y_pred,thresh):
    auc = roc_auc_score(y_actual,y_pred)
    accuracy = accuracy_score(y_actual,(y_pred>thresh))
    recall = recall_score(y_actual,(y_pred>thresh))
    precision = precision_score(y_actual,(y_pred>thresh))
    specificity = dsa_calcula_especificidade(y_actual,y_pred,thresh)

    print(f'AUC: {auc:.3f}')
    print(f'Acurácia: {accuracy:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'Precisão: {precision:.3f}')
    print(f'Especificidade: {specificity:.3f}')

    return auc,accuracy,recall,precision,specificity
thresh = 0.5

lr1 = LogisticRegression()
modelo_v1 = lr1.fit(X_treino_tf,y_treino)

y_train_preds = modelo_v1.predict_proba(X_treino_tf)[:,1]
y_valid_preds = modelo_v1.predict_proba(X_valid_tf)[:,1]

print('Modelo de Regressão Logística (Sem Otimização de Hiperparâmetros)\n')

print('Treinamento:\n')
lr1_train_auc, lr1_train_accuracy, lr1_train_recall, lr1_train_precision, lr1_train_specificity = dsa_print_report(y_treino, 
                                                                                                                   y_train_preds, 
                                                                                                                   thresh)

print('Validação:\n')
lr1_valid_auc, lr1_valid_accuracy, lr1_valid_recall, lr1_valid_precision, lr1_valid_specificity = dsa_print_report(y_valid, 
                                                                                                                   y_valid_preds, 
                                                                                                                   thresh)


lr2 = LogisticRegression(random_state=142,solver='liblinear')
modelo_v2 = lr2.fit(X_treino_tf,y_treino)

y_train_preds = modelo_v2.predict_proba(X_treino_tf)[:,1]
y_valid_preds = modelo_v2.predict_proba(X_valid_tf)[:,1]

print('Modelo de Regressão Logística (Com Otimização de Hiperparâmetros)\n')

print('Treinamento:\n')
lr2_train_auc, lr2_train_accuracy, lr2_train_recall, lr2_train_precision, lr2_train_specificity = dsa_print_report(y_treino, 
                                                                                                                   y_train_preds, 
                                                                                                                   thresh)

print('Validação:\n')
lr2_valid_auc, lr2_valid_accuracy, lr2_valid_recall, lr2_valid_precision, lr2_valid_specificity = dsa_print_report(y_valid, 
                                                                                                                   y_valid_preds, 
                                                                                                                   thresh)

nb = GaussianNB()
modelo_v3 = nb.fit(X_treino_tf,y_treino)
y_train_preds = modelo_v3.predict_proba(X_treino_tf)[:,1]
y_valid_preds = modelo_v3.predict_proba(X_valid_tf)[:,1]

print('Modelo Naive Bayes:\n')

print('Treinamento:\n')
nb_train_auc, nb_train_accuracy, nb_train_recall, nb_train_precision, nb_train_specificity = dsa_print_report(y_treino, 
                                                                                                              y_train_preds, 
                                                                                                              thresh)

print('Validação:\n')
nb_valid_auc, nb_valid_accuracy, nb_valid_recall, nb_valid_precision, nb_valid_specificity = dsa_print_report(y_valid, 
                                                                                                              y_valid_preds, 
                                                                                                              thresh)

xgbc = XGBClassifier()
modelo_v4 = xgbc.fit(X_treino_tf,y_treino)
y_train_preds = modelo_v4.predict_proba(X_treino_tf)[:,1]
y_valid_preds = modelo_v4.predict_proba(X_valid_tf)[:,1]
print('Modelo XGBoost:\n')

print('Treinamento:\n')
xgbc_train_auc, xgbc_train_accuracy, xgbc_train_recall, xgbc_train_precision, xgbc_train_specificity = dsa_print_report(y_treino, 
                                                                                                                        y_train_preds, 
                                                                                                                        thresh)

print('Validação:\n')
xgbc_valid_auc, xgbc_valid_accuracy, xgbc_valid_recall, xgbc_valid_precision, xgbc_valid_specificity = dsa_print_report(y_valid, 
                                                                                                                        y_valid_preds, 
                                                                                                                        thresh)

rfc = RandomForestClassifier()
modelo_v5 = rfc.fit(X_treino_tf,y_treino)
y_train_preds = modelo_v5.predict_proba(X_treino_tf)[:,1]
y_valid_preds = modelo_v5.predict_proba(X_valid_tf)[:,1]
print('Modelo Random Forest Classifier:\n')
print('Treinamento:\n')
rfc_train_auc, rfc_train_accuracy, rfc_train_recall, rfc_train_precision, rfc_train_specificity = dsa_print_report(y_treino, 
                                                                                                                   y_train_preds, 
                                                                                                                   thresh)

print('Validação:\n')
rfc_valid_auc, rfc_valid_accuracy, rfc_valid_recall, rfc_valid_precision, rfc_valid_specificity = dsa_print_report(y_valid, 
                                                                                                                   y_valid_preds, 
                                                                                                                   thresh)

df_results = pd.DataFrame({'classificador': ['RL1','RL1','RL2','RL2','NB','NB','XGB','XGB','RFC','RFC'],
                           'data_set':['treino','valid'] * 5,
                           'auc': [lr1_train_auc,
                                   lr1_valid_auc,
                                   lr2_train_auc,
                                   lr2_valid_auc,
                                   nb_train_auc,
                                   nb_valid_auc,
                                   xgbc_train_auc,
                                   xgbc_valid_auc,
                                   rfc_train_auc,
                                   rfc_valid_auc],
                            'accuracy': [lr1_train_accuracy,
                                         lr1_valid_accuracy,
                                         lr2_train_accuracy,
                                         lr2_valid_accuracy,
                                         nb_train_accuracy,
                                         nb_valid_accuracy,
                                         xgbc_train_accuracy,
                                         xgbc_valid_accuracy,
                                         rfc_train_accuracy,
                                         rfc_valid_accuracy],
                            'recall': [lr1_train_recall,
                                       lr1_valid_recall,
                                       lr2_train_recall,
                                       lr2_valid_recall,
                                       nb_train_recall,
                                       nb_valid_recall,
                                       xgbc_train_recall,
                                       xgbc_valid_recall,
                                       rfc_train_recall,
                                       rfc_valid_recall],
                            'precision': [lr1_train_precision,
                                       lr1_valid_precision,
                                       lr2_train_precision,
                                       lr2_valid_precision,
                                       nb_train_precision,
                                       nb_valid_precision,
                                       xgbc_train_precision,
                                       xgbc_valid_precision,
                                       rfc_train_precision,
                                       rfc_valid_precision],
                            'specificity': [lr1_train_specificity,
                                       lr1_valid_specificity,
                                       lr2_train_specificity,
                                       lr2_valid_specificity,
                                       nb_train_specificity,
                                       nb_valid_specificity,
                                       xgbc_train_specificity,
                                       xgbc_valid_specificity,
                                       rfc_train_specificity,
                                       rfc_valid_specificity]})
print(df_results)

print(df_results[df_results['data_set'] == 'valid'].sort_values(by='auc',ascending=False))

sns.set_style('whitegrid')
plt.figure(figsize=(16,8))
ax = sns.barplot(x='classificador',y='auc',hue='data_set',data=df_results)
ax.set_xlabel('Classificador',fontsize=15)
ax.set_ylabel('AUC',fontsize=15)
ax.tick_params(labelsize=15)
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.,fontsize=15)
plt.show()

param_grid = {
    'n_estimators': [100,200,300],
    'max_depth': [None,10,20,30],
    'min_samples_split': [2,5,10],
    'min_samples_leaf': [1,2,4]
}

classificador = RandomForestClassifier()
grid_search = GridSearchCV(estimator=classificador,param_grid=param_grid,cv=5,scoring='roc_auc',verbose=2,n_jobs=-1)
modelo_v5_otimizado = grid_search.fit(X_treino_tf,y_treino)
print('Melhores Hiperparâmetros:',modelo_v5_otimizado.best_params_)
y_train_preds = modelo_v5_otimizado.predict_proba(X_treino_tf)[:,1]
y_valid_preds = modelo_v5_otimizado.predict_proba(X_valid_tf)[:,1]

print('Modelo Random Forest Classifier com Otimização de Hiperparâmetros e Validação Cruzada:\n')
print('Treinamento:\n')
rfc_train_auc, rfc_train_accuracy, rfc_train_recall, rfc_train_precision, rfc_train_specificity = dsa_print_report(y_treino, 
                                                                                                                   y_train_preds, 
                                                                                                                   thresh)

print('Validação:\n')
rfc_valid_auc, rfc_valid_accuracy, rfc_valid_recall, rfc_valid_precision, rfc_valid_specificity = dsa_print_report(y_valid, 
                                                                                                                   y_valid_preds, 
                                                                                                                   thresh)

pickle.dump(modelo_v5, open('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_2/modelos/melhor_modelo.pkl','wb'),protocol=4)

best_model = pickle.load(open('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_2/modelos/melhor_modelo.pkl','rb'))
cols_input = pickle.load(open('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_2/dados/colunas_entrada.sav','rb'))
scaler = pickle.load(open('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_2/dados/scaler.sav','rb'))

df_treino = pd.read_csv('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_2/dados/dados_treino_final.csv')
df_valid = pd.read_csv('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_2/dados/dados_valid.csv')
df_test = pd.read_csv('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_2/dados/dados_teste.csv')

X_train = df_treino[cols_input].values
X_valid = df_valid[cols_input].values
X_test = df_test[cols_input].values

y_train = df_treino['LABEL_TARGET'].values
y_valid = df_valid['LABEL_TARGET'].values
y_test = df_test['LABEL_TARGET'].values

X_train_tf = scaler.transform(X_train)
X_valid_tf = scaler.transform(X_valid)
X_test_tf = scaler.transform(X_test)

y_train_preds = best_model.predict_proba(X_train_tf)[:,1]
y_valid_preds = best_model.predict_proba(X_valid_tf)[:,1]
y_test_preds = best_model.predict_proba(X_test_tf)[:,1]

print('Treinamento:\n')
rfc_train_auc, rfc_train_accuracy, rfc_train_recall, rfc_train_precision, rfc_train_specificity = dsa_print_report(y_treino, 
                                                                                                                   y_train_preds, 
                                                                                                                   thresh)

print('Validação:\n')
rfc_valid_auc, rfc_valid_accuracy, rfc_valid_recall, rfc_valid_precision, rfc_valid_specificity = dsa_print_report(y_valid, 
                                                                                                                   y_valid_preds, 
                                                                                                                   thresh)

print('Teste:\n')
rfc_test_auc, rfc_test_accuracy, rfc_test_recall, rfc_test_precision, rfc_test_specificity = dsa_print_report(y_test, 
                                                                                                                   y_test_preds, 
                                                                                                                   thresh)

# Calcula a curva ROC nos dados de treino
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_preds)
auc_train = roc_auc_score(y_train, y_train_preds)

# Calcula a curva ROC nos dados de validação
fpr_valid, tpr_valid, thresholds_valid = roc_curve(y_valid, y_valid_preds)
auc_valid = roc_auc_score(y_valid, y_valid_preds)

# Calcula a curva ROC nos dados de teste
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_preds)
auc_test = roc_auc_score(y_test, y_test_preds)

# Plot
plt.figure(figsize=(16,10))
plt.plot(fpr_train, tpr_train, 'r-', label = 'AUC em Treino: %.3f'%auc_train)
plt.plot(fpr_valid, tpr_valid, 'b-', label = 'AUC em Validação: %.3f'%auc_valid)
plt.plot(fpr_test, tpr_test, 'g-', label = 'AUC em Teste: %.3f'%auc_test)
plt.plot([0,1],[0,1],'k--')
plt.xlabel('Taxa de Falso Positivo')
plt.ylabel('Taxa de Verdadeiro Positivo')
plt.legend()
plt.show()

novo_cliente = pd.read_csv('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/Projeto_2/dados/novo_cliente.csv')
novo_cliente_scaled = scaler.transform(novo_cliente)
print(novo_cliente_scaled)
print(best_model.predict_proba(novo_cliente_scaled))
print(best_model.predict(novo_cliente_scaled))