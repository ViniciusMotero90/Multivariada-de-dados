import numpy as np
import pandas as pd

df_dsa = pd.read_csv('C:/Users/Vinícius/Desktop/DSA/Multivariada de dados/capitulo_15/dataset.csv')

df_numeric =df_dsa.select_dtypes(include=[np.number])

df_numeric = df_numeric.fillna(method='ffill')

data_matrix = df_numeric.to_numpy()
print(data_matrix.shape)

def my_dsa_PCA(input):
    mean = np.mean(input,axis=0)
    normalised_input = input - mean
    normalised_input_transpose = normalised_input.T
    num_of_samples = input.shape[0]
    cov_mat = np.dot(normalised_input_transpose,normalised_input) / num_of_samples
    print('\nMatriz de Covariância:')
    print(cov_mat)

    eigenvalues,eigenvectors = np.linalg.eig(cov_mat)
    print('\nAutovalores:')
    print(eigenvalues)
    print('\nAutovetores:')
    print(eigenvectors)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:,sorted_indices]
    sorted_eigenvalues = eigenvalues[sorted_indices]

    print('\nComponentes Principais:')
    print(sorted_eigenvectors)

    pca_output = np.dot(normalised_input,sorted_eigenvectors)

    print('\nNovos Dados com Dimensão Reduzida:')
    print(pca_output)

    return pca_output, sorted_eigenvalues, sorted_eigenvectors, cov_mat

pca_output, eigenval, principal_component, cov_matrix = my_dsa_PCA(data_matrix)

addition = (eigenval[0]/sum(eigenval))+(eigenval[1]/sum(eigenval))+(eigenval[2]/sum(eigenval))+(eigenval[3]/sum(eigenval))

percentage = addition * 100
print(percentage)
print("\nQuatro Componentes Principais Explicam aproximadamente 77% da Variação nos Dados")