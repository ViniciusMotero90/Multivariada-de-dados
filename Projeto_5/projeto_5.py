import pandas as pd
import numpy as np
import factor_analyzer
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import matplotlib.pyplot as plt

df_dsa = pd.read_csv('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/projteo_5/dataset.csv')

chi_square_value, p_value = calculate_bartlett_sphericity(df_dsa)
print('Estatística do Teste:',chi_square_value,'\nValor-p:',p_value)

kmo_all, kmo_model = calculate_kmo(df_dsa)
print('KMO Global:',kmo_model)
print('KMO Por Variável:',kmo_all)

fa = FactorAnalyzer(rotation='varimax')
fa.fit(df_dsa)

eigen_values, vectors = fa.get_eigenvalues()
plt.scatter(range(1,df_dsa.shape[1]+1),eigen_values)
plt.plot(range(1,df_dsa.shape[1]+1),eigen_values)
plt.title("Gráfico de Cotovelo")
plt.xlabel('Fatores')
plt.ylabel('Autovalor')
plt.grid()
plt.show()

fa = FactorAnalyzer(n_factors=2,rotation='varimax')
fa.fit(df_dsa)

loadings = fa.loadings_
print(loadings)
print(fa.get_communalities())