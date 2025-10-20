import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from itertools import cycle
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from itertools import cycle
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
# In[1] Configurações globais 

# Colocar o ficheiro em Excell
sns.set()
excel_path = r"C:\Users\Andreia\Desktop\beetroot\Data\Beetroot.xlsx"

# In[2] Data Understanding:

# Ler o DataFrame
df = pd.read_excel(excel_path);

# Obter informações sobre o DataFrame
print(df.info());

# Obter Dados do DataFrame
print("\n\nColumns: ");
for column in df.columns:
        print(column);

print(df.head());

print("\n Estatísticas Descritivas do DataFrame:")
print("\n", df.describe());

# In[3] Data Preparation:
    
# Remover linhas com valores NaN 
df = df.dropna()

# Valores duplicados
df = df.drop_duplicates()

# Substituir os zeros pela média das colunas
for coluna in df.columns:
    media_coluna = df[coluna][df[coluna] != 0].mean()
    df[coluna] = df[coluna].replace(0, media_coluna)

# Identificar e remover outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < (Q1 - 1.1 * IQR)) | (df > (Q3 + 1.1 * IQR))).any(axis=1)
df_sem_outliers = df[~outliers]

# Exibir estatísticas dos dados sem outliers
print("\nEstatísticas sem outliers:")
print(df_sem_outliers.describe())

# In[] Boxplot

plt.figure(figsize=(12, 8))

# Boxplot para as variáveis 'Solvent', 'Order', 'TPC' e 'ANT'
plt.subplot(1, 2, 1)
sns.boxplot(data=df_sem_outliers[['Solvent', 'Order', 'TPC', 'ANT']])
plt.title('Boxplot de Solvent, Order, TPC e ANT (Sem outliers)')
plt.xlabel('Variáveis')

# Calcular estatísticas adicionais para cada variável
stats_sem_outliers = df_sem_outliers[['Solvent', 'Order', 'TPC', 'ANT']].describe()

# Imprimir as estatísticas
print("Estatísticas adicionais para cada variável:\n")
print(stats_sem_outliers)

# Boxplot para as variáveis 'Vm-ratio' e 'AOA'
plt.subplot(1, 2, 2)
sns.boxplot(data=df_sem_outliers[['Vm-ratio', 'Time', 'AOA']])
plt.title('Boxplot de Vm-ratio, Time e AOA (Sem outliers)')
plt.xlabel('Variáveis')

plt.show()

# Calcular estatísticas adicionais para cada variável
stats_vm_time_aoa = df_sem_outliers[['Vm-ratio', 'Time', 'AOA']].describe()

# Imprimir as estatísticas
print("Estatísticas adicionais para cada variável :\n")
print(stats_vm_time_aoa)

# In[] Histograma

# Configurações do gráfico
plt.figure(figsize=(18, 12))
plt.style.use('seaborn-darkgrid')

# Plotar histogramas para as colunas de entrada 'Solvent', 'Vm-ratio', 'Order', 'Time'
for i, coluna in enumerate(['Solvent', 'Vm-ratio', 'Order', 'Time']):
    plt.subplot(2, 4, i + 1)
    sns.histplot(df[coluna], kde=True)
    plt.xlabel(coluna + ' (Input)')

# Plotar histogramas para as colunas de saída 'TPC', 'AOA' e 'ANT'
for i, coluna in enumerate(['TPC', 'AOA', 'ANT']):
    plt.subplot(2, 4, i + 5)
    sns.histplot(df[coluna], kde=True)
    plt.xlabel(coluna + ' (Output)')

plt.tight_layout()
plt.show()

# In[]] Matriz de Correlação

correlation_matrix = df.corr()

# Plt da matriz de correlação  usando o heatmap

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de Correlação')
plt.show()


# In[] Gráfico de Dispersão para Variáveis de Entrada versus Variáveis de Saída


# Definir paleta de cores
palette = cycle(sns.color_palette())

# Configurações do gráfico
plt.figure(figsize=(18, 12))
plt.style.use('seaborn-darkgrid')

# Variáveis de entrada
variaveis_entrada = ['Solvent', 'Vm-ratio', 'Order', 'Time']
variaveis_saida= ['TPC','AOA','ANT']

# Plotar gráfico de dispersão para cada variável de entrada versus cada variável de saída
for i, entrada in enumerate(variaveis_entrada):
    for j, saida in enumerate(variaveis_saida):
        plt.subplot(len(variaveis_entrada), len(variaveis_saida), i*len(variaveis_saida)+j+1)
        cor = next(palette)  # Obtém a próxima cor da paleta
        plt.scatter(df[entrada], df[saida], label=f'{entrada} vs {saida}', color=cor, alpha=0.6)
        plt.xlabel(entrada)
        plt.ylabel(saida)
        plt.title(f'{entrada} vs {saida}')
        plt.legend()

plt.tight_layout()


plt.show()



# In[] Variaveis categóricas 

one_hot_encoder = OneHotEncoder()

# Ajustar e transformar os dados categóricos
encoded_data = one_hot_encoder.fit_transform(df[['Order', 'Time', 'Vm-ratio', 'Solvent']])

# Converter a saída em um DataFrame pandas e obter os nomes das colunas
encoded_df = pd.DataFrame(encoded_data.toarray(), columns=one_hot_encoder.get_feature_names_out(['Order', 'Time', 'Vm-ratio', 'Solvent']))

# Concatenar o DataFrame codificado com o DataFrame original
df_= pd.concat([df, encoded_df], axis=1)

# Remover as colunas originais categóricas
df_.drop(columns=['Order', 'Time', 'Vm-ratio', 'Solvent'],inplace=True)

# In[] Normalização 

# Especificar as variáveis de entrada (X) e saída (y)
X_cols = ['Solvent', 'Order', 'Time', 'Vm-ratio']  # Defina as variáveis de entrada
y_cols = ['TPC', 'AOA', 'ANT']  # Defina as variáveis de saída

# Separar as variáveis de entrada e saída
X = df[X_cols]
y = df[y_cols]

# Normalizar os dados de entrada
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# In[] Modeling

# Divisão dos dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Inicializar o modelo RandomForestRegressor
model_rf_tuned = RandomForestRegressor()

# Realizar a busca em grade para encontrar os melhores hiperparâmetros
grid_search = GridSearchCV(estimator=model_rf_tuned, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Obter os melhores hiperparâmetros encontrados
best_params = grid_search.best_params_
print("Melhores hiperparâmetros encontrados para RandomForestRegressor:", best_params)



# 2. PCA com Normalização
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_normalized)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

model_rf_pca = RandomForestRegressor()
model_rf_pca.fit(X_train_pca, y_train_pca)
y_pred_rf_pca = model_rf_pca.predict(X_test_pca)



# 3. Regressão Linear com Normalização
y_linear = df[y_cols[2]]  # Escolher uma das saídas para a regressão linear
X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(X_normalized, y_linear, test_size=0.2, random_state=42)

model_linear = LinearRegression()
model_linear.fit(X_train_linear, y_train_linear)
y_pred_linear = model_linear.predict(X_test_linear)



# In[] Evoluation

# Avaliar o modelo ajustado(RandomForest)
y_pred_rf_tuned = grid_search.predict(X_test)
mse_rf_tuned = mean_squared_error(y_test, y_pred_rf_tuned)
rmse_rf_tuned = mean_squared_error(y_test, y_pred_rf_tuned, squared=False)
r2_rf_tuned = r2_score(y_test, y_pred_rf_tuned)
mae_rf_tuned = mean_absolute_error(y_test, y_pred_rf_tuned)
mape_rf_tuned = (mean_absolute_error(y_test, y_pred_rf_tuned) / y_test.mean().mean()) * 100

print("\nAvaliação para RandomForestRegressor com Tuning de Hiperparâmetros:")
print("MSE:", mse_rf_tuned)
print("RMSE:", rmse_rf_tuned)
print("R²:", r2_rf_tuned)
print("MAE:", mae_rf_tuned)
print("MAPE:", mape_rf_tuned)

#PCA
mse_rf_pca = mean_squared_error(y_test_pca, y_pred_rf_pca)
rmse_rf_pca = mean_squared_error(y_test_pca, y_pred_rf_pca, squared=False)
r2_rf_pca = r2_score(y_test_pca, y_pred_rf_pca)
mae_rf_pca = mean_absolute_error(y_test_pca, y_pred_rf_pca)
mape_rf_pca = (mean_absolute_error(y_test_pca, y_pred_rf_pca) / y_test_pca.mean().mean()) * 100

print("\nAvaliação do PCA com normalização :")
print("MSE:", mse_rf_pca)
print("RMSE:", rmse_rf_pca)
print("R²:", r2_rf_pca)
print("MAE:", mae_rf_pca)
print("MAPE:", mape_rf_pca)

#Regressao Linear
mse_linear = mean_squared_error(y_test_linear, y_pred_linear)
rmse_linear = mean_squared_error(y_test_linear, y_pred_linear, squared=False)
r2_linear = r2_score(y_test_linear, y_pred_linear)
mae_linear = mean_absolute_error(y_test_linear, y_pred_linear)
mape_linear = (mean_absolute_error(y_test_linear, y_pred_linear) / y_test_linear.mean()) * 100

print("\nRegressão Linear com Normalização:")
print("MSE:", mse_linear)
print("RMSE:", rmse_linear)
print("R²:", r2_linear)
print("MAE:", mae_linear)
print("MAPE:", mape_linear)

# Definir métricas para RandomForestRegressor com Normalização
metrics_rf_normalized = {
    'MSE': mse_rf_tuned,
    'RMSE': rmse_rf_tuned,
    'R²': r2_rf_tuned,
    'MAE': mae_rf_tuned,
    'MAPE': mape_rf_tuned
}

# Definir métricas para RandomForestRegressor com PCA
metrics_rf_pca = {
    'MSE': mse_rf_pca,
    'RMSE': rmse_rf_pca,
    'R²': r2_rf_pca,
    'MAE': mae_rf_pca,
    'MAPE': mape_rf_pca
}

# Definir métricas para Regressão Linear
metrics_linear = {
    'MSE': mse_linear,
    'RMSE': rmse_linear,
    'R²': r2_linear,
    'MAE': mae_linear,
    'MAPE': mape_linear
}


# Definir a lista de métricas
metrics = ['MSE', 'RMSE', 'R²', 'MAE', 'MAPE']

# Plotar as métricas
plt.figure(figsize=(12, 8))


# Definir a largura das barras
width = 0.25
# Definir as posições das barras
positions_rf_normalized = np.arange(len(metrics))
positions_rf_pca = [pos + width for pos in positions_rf_normalized]
positions_linear = [pos + 2 * width for pos in positions_rf_normalized]

# Plotar as barras para RandomForestRegressor com Normalização
for i, metric in enumerate(metrics):
    plt.bar(positions_rf_normalized[i], metrics_rf_normalized[metric], width, label=f'{metric} (RandomForest)', color='blue')
    plt.text(positions_rf_normalized[i], metrics_rf_normalized[metric] + 0.01, str(round(metrics_rf_normalized[metric], 2)), ha='center', va='bottom')

# Plotar as barras para RandomForestRegressor com PCA
for i, metric in enumerate(metrics):
    plt.bar(positions_rf_pca[i], metrics_rf_pca[metric], width, label=f'{metric} (PCA)', color='orange')
    plt.text(positions_rf_pca[i], metrics_rf_pca[metric] + 0.01, str(round(metrics_rf_pca[metric], 2)), ha='center', va='bottom')

# Plotar as barras para Regressão Linear
for i, metric in enumerate(metrics):
    plt.bar(positions_linear[i], metrics_linear[metric], width, label=f'{metric} (Regressão Linear)', color='green')
    plt.text(positions_linear[i], metrics_linear[metric] + 0.01, str(round(metrics_linear[metric], 2)), ha='center', va='bottom')

# Adicionar rótulos aos eixos e título ao gráfico
plt.xlabel('Métricas')
plt.ylabel('Valor')
plt.title('Comparação das Métricas de Avaliação dos Modelos')
plt.xticks([pos + width for pos in positions_rf_normalized], metrics) 
plt.legend()

# Exibir o gráfico
plt.show()