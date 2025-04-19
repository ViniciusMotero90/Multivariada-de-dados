import scipy
import statsmodels
import numpy as np
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
from scipy.stats import f
from numpy.linalg import inv, det
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

np.random.seed(0)

grupo_1 = np.random.multivariate_normal([170,60],[[10,2],[2,5]],50)
grupo_2 = np.random.multivariate_normal([175,65],[[10,2],[2,5]],50)

print(grupo_1[1:5])
print(grupo_2[1:5])

print(np.mean(grupo_1,axis=0))
print(np.mean(grupo_2,axis=0))

def hotelling_t2_test(group1,group2):
    mean1 = np.mean(group1,axis=0)

    mean2 = np.mean(group2,axis=0)

    n1,n2 = len(group1), len(group2)

    cov1 = np.cov(group1.T)

    cov2 = np.cov(group2.T)

    pooled_cov = ((n1 - 1) * cov1 + (n2 - 1) * cov2) / (n1 + n2 - 2)

    mean_diff = mean1 - mean2

    t2_stat = n1 * n2 / (n1 + n2) * mean_diff.dot(inv(pooled_cov)).dot(mean_diff)

    df1 = len(mean1)

    df2 = n1 + n2 - df1 - 1
    f_stat = t2_stat * (df2 / (n1 + n2 - 2)) / df1

    p_value = 1 - f.cdf(f_stat,df1,df2)
    return t2_stat, p_value

t2_stat, p_value = hotelling_t2_test(grupo_1,grupo_2)
print("Estatística T²:",t2_stat)
print("Valor-p:",p_value)

np.random.seed(0)
group = np.repeat(['A','B','C'],20)
data1 = np.random.normal(0,1,60) + (group == 'B') * 1 + (group == 'C') * 2
data2 = np.random.normal(0,1,60) + (group == 'B') * 1.5 + (group == 'C') * -1
df_dsa = pd.DataFrame({'Grupo':group,'DV1':data1,'DV2':data2})

print(df_dsa)

fig, axs = plt.subplots(ncols=2)
sns.boxplot(data=df_dsa,x='Grupo',y='DV1',hue=df_dsa.Grupo.tolist(),ax=axs[0])
sns.boxplot(data=df_dsa,x='Grupo',y='DV2',hue=df_dsa.Grupo.tolist(),ax=axs[1])
plt.show()

moav = MANOVA.from_formula('DV1 + DV2 ~ Grupo', data=df_dsa)
print(moav.mv_test())

X = df_dsa[["DV1","DV2"]]
y = df_dsa["Grupo"]

modelo = lda().fit(X=X,y=y)
print(modelo.priors_)
print(modelo.means_)
print(modelo.scalings_)
print(modelo.explained_variance_ratio_)

X_new = pd.DataFrame(lda().fit(X=X,y=y).transform(X),columns=["lda1","lda2"])
X_new["Grupo"] = df_dsa["Grupo"]
sns.scatterplot(data=X_new,x='lda1',y='lda2',hue=df_dsa.Grupo.tolist())
plt.show()