# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %% [markdown]
# # Análise dos dados

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# df_bmi_age contem o bmi, a idade e o estado (não diabético/diabético) de  3080 individuos identificados pelo ID
df_bmi_age = pd.read_csv("/home/jgcarvalho/colaboracoes/paulao/diabetes_snp/development/data_26_09_2019/T2D_BMI_AGE.csv").dropna()
df_train = pd.read_csv("../datasets/visits_new_train_positivo_1000_random_0.csv")
df_test = pd.read_csv("../datasets/visits_new_test_positivo_1000_random_0.csv")

# %% [markdown]
# #### Análise geral da relação entre BMI, AGE e T2D.
# Observações:
# - Na amostra, a distribuição de idades dos diabéticos indica indivíduos mais velhos
# - A distribuição de BMI indica valores um pouco mais elevados para os diabéticos. No entanto, não temos certeza da relevância de tal dado (ver gráfico a seguir).

# %%
sns.pairplot(df_bmi_age[["AGE", "BMI", "T2D"]], hue="T2D", markers='+');

# %% [markdown]
# Indivíduos mais jovens, abaixo de 30 anos, demonstram uma tendência de terem BMI menor. Como indivíduos nessa faixa etária são bem mais numerosos no grupo de diabéticos, eles poderiam estar distorcendo a comparação das distribuições de BMI entre os grupos.

# %%
# gráfico abaixo: não diabéticos em azul, diabéticos em laranja
diabetics = df_bmi_age[df_bmi_age.T2D > 0.5]
n_diabetics = df_bmi_age[df_bmi_age.T2D < 0.5]
ax = sns.kdeplot(n_diabetics.BMI, n_diabetics.AGE, cmap="Blues", shade=True, shade_lowest=False)
ax = sns.kdeplot(diabetics.BMI, diabetics.AGE, cmap="Oranges", shade=False, shade_lowest=False)

# %%
sns.jointplot(x="BMI", y="AGE", data=df_bmi_age, kind="kde")

# %% [markdown]
# Além dessa "incerteza" entre as distribuições de BMI, não recebemos o BMI medido em cada visita (informação temporal do paciente). Consequentemente, no momento atual, os dados de BMI não são utilizado no modelo.