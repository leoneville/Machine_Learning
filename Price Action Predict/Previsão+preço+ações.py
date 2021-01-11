
# coding: utf-8

# In[7]:

#Importando as bibliotecas
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[12]:

#Lendo o arquivo de ações
#Lendo csv
df=pd.read_csv( 'df_all_bovespa.csv', delimiter=',' )
df


# In[16]:

#Previsao Itau
df_itau = df[df['sigla_acao'] == 'ITUB4']
df_itau.head(10)


# In[17]:

#Verificar o tipo do arquivo
df_itau.dtypes


# In[19]:

#Mudar o tipo data
df_itau['data_pregao'] = pd.to_datetime(df_itau['data_pregao'], format='%Y-%m-%d')


# In[20]:

df_itau.dtypes


# In[21]:

df_itau.tail()


# In[22]:

#Criando novos campos de medias móveis
df_itau['mm5d'] = df_itau['preco_fechamento'].rolling(5).mean()
df_itau['mm21d'] = df_itau['preco_fechamento'].rolling(21).mean()


# In[23]:

df_itau.head(7)


# In[24]:

#Empurrando para frente os valores das ações
df_itau['preco_fechamento'] = df_itau['preco_fechamento'].shift(-1)

df_itau.head()


# In[25]:

#Retirando os dados nulos
df_itau.dropna(inplace=True)
df_itau


# In[27]:

#Verificando quantidade de linhas
qtd_linhas = len(df_itau)
qtd_linhas_treino = qtd_linhas - 700
qtd_linhas_teste = qtd_linhas - 15

qtd_linhas_validacao = qtd_linhas_treino - qtd_linhas_teste

info = (
    f"linhas treino= 0:{qtd_linhas_treino}"
    f" linhas teste= {qtd_linhas_treino}:{qtd_linhas_teste}"
    f" linhas validacao= {qtd_linhas_teste}:{qtd_linhas}"
)

info


# In[29]:

#Reindexando o Data Frame
df_itau = df_itau.reset_index(drop=True)
df_itau


# In[30]:

#Separando as features e labels
features = df_itau.drop(['sigla_acao', 'nome_acao', 'data_pregao', 'preco_fechamento'], 1)
labels = df_itau['preco_fechamento']


# In[31]:

#Escolhendo as melhores features com KBest

features_list = ('preco_abertura', 'qtd_total_negociado', 'volume_total_negociado', 'mm5d', 'mm21d')

k_best_features = SelectKBest(k='all')
k_best_features.fit_transform(features, labels)
k_best_features_scores = k_best_features.scores_
raw_pairs = zip(features_list[1:], k_best_features_scores)
ordered_pairs = list(reversed(sorted(raw_pairs, key=lambda x: x[1])))

k_best_features_final = dict(ordered_pairs[:15])
best_features = k_best_features_final.keys()
print ('')
print ("Melhores Features:")
print(k_best_features_final)


# In[32]:

#Separando as features escolhidas
features = df_itau.drop(['sigla_acao', 'nome_acao', 'data_pregao', 'preco_fechamento', 'preco_abertura', 'mm21d'], 1)


# In[33]:

#Normalizando os dados de entrada(features)

#Gerando o novo padrão
scaler = MinMaxScaler().fit(features)
features_scale = scaler.transform(features)

print('features: ',features_scale.shape)
print(features_scale) # Normalizando os dados de entrada(features)


# In[34]:

#Separa os dados de treino teste e validacao
X_train = features_scale[:qtd_linhas_treino]
X_test = features_scale[qtd_linhas_treino:qtd_linhas_teste]

y_train = labels[:qtd_linhas_treino]
y_test = labels[qtd_linhas_treino:qtd_linhas_teste]

print(len(X_train), len(y_train))
print(len(X_test), len(y_test))


# In[38]:

#treinamento usando regressao linear
lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
cd = r2_score(pred, y_test)

f'Coeficiente de determinação: {cd * 100:.2f}'


# In[42]:

#Rede Neural
rn = MLPRegressor(max_iter=2000)

rn.fit(X_train, y_train)
pred = rn.predict(X_test)

cd = rn.score(X_test, y_test)

f'Coeficiente de determinação:{cd * 100:.2f}'


# In[44]:

#Rede Neural com ajuste hyper parameters

rn = MLPRegressor()

parameter_space = {
        'hidden_layer_sizes': [(i,) for i in list(range(1, 21))],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }

search = GridSearchCV(rn, parameter_space, n_jobs=-1, cv=5)

search.fit(X_train, y_train)
clf = search.best_estimator_
pred = search.predict(X_test)

cd = search.score(X_test, y_test)

f'Coeficiente de determinação: {cd * 100:.2f}'


# In[56]:

#Executando a previsão Linear Regression

previsao = features_scale[qtd_linhas_teste:qtd_linhas]

data_pregao_full = df_itau['data_pregao']
data_pregao = data_pregao_full[qtd_linhas_teste:qtd_linhas]

res_full = df_itau['preco_fechamento']
res = res_full[qtd_linhas_teste:qtd_linhas]

pred = lr.predict(previsao)

df = pd.DataFrame({'data_pregao':data_pregao, 'real':res, 'previsao':pred})
df['real'] = df['real'].shift(+1)

df.set_index('data_pregao', inplace=True)

print(df)


# In[57]:

#grafico
plt.figure(figsize=(16,8))
plt.title('Preço das ações')
plt.plot(df['real'],label="real",color='blue', marker='o')
plt.plot(df['previsao'],label="previsao",color='red', marker='o')
plt.xlabel('Data pregão')
plt.ylabel('Preço de Fechamento')
leg = plt.legend()


# In[58]:

pred = search.predict(previsao)

df = pd.DataFrame({'data_pregao':data_pregao, 'real':res, 'previsao':pred})
df['real'] = df['real'].shift(+1)

df.set_index('data_pregao', inplace=True)

print(df)


# In[59]:

#grafico
plt.figure(figsize=(16,8))
plt.title('Preço das ações')
plt.plot(df['real'],label="real",color='blue', marker='o')
plt.plot(df['previsao'],label="previsao",color='red', marker='o')
plt.xlabel('Data pregão')
plt.ylabel('Preço de Fechamento')
leg = plt.legend()


# In[ ]:



