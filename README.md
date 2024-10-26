# O objetivo desse projeto prático é utilizar vários modelos de classificação para detectar se uma pessoa tem câncer de mama  ou não.

*Tecnologias e bibliotecas utilizadas*: Python, Pandas, NumPy, Matplotlib, SKLEARN, XGBoost, LightGBM e CatBoost.

*Modelos utilizados*: Naive Bayes, SVM, Regressão Logística, KNN, Árvores de Decisão, Random Forest, XGBoost, LighGBM e CatBoost.

## Base de dados inicial

![image](https://github.com/user-attachments/assets/8cf01d5c-d60d-4764-8620-fe553f9ea739)


## Tratamento dos dados

Primeiro tratamento na base de dados é para remover tudo que não for númerico, para fazer isso usamos essa solução:

``` df3.replace({'[^0-9]': ''}, regex = True) ```

Depois uma função para ajustar os formatos das variáveis númericas, para todas ficarem no mesmo padrão:

```
def ajustar_formato(x):

    if isinstance(x, str) and '.' not in x:

        return f'{int(x) / 100:.2f}'

    else:

        return x
```

Logo após foi feito uma remoção de uma coluna desnecessária para o nosso modelo e remoção de NaN: 

```
df4.drop(['Unnamed: 32'], axis = 1)
df4.dropna
```

## Seleção de previsores e alvo do modelo

```
previsores = df4.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1).values #Seleciona todas variáveis menos 'id' e 'diagnosis'
alvos = df4['diagnosis'].values #Seleciona apenas 'diagnosis' como alvo

```

## Escalonamento para melhorar precisão em certos algoritmos

```
scaler = StandardScaler()
previsores_esc = scaler.fit_transform(previsores)

```

Base de treino e teste utilizando os previsores e os alvos

```
x_treino, x_teste, y_treino, y_teste = train_test_split(previsores, alvos, test_size = 0.3, random_state = 0)
```

Dito isto, o objetivo deste projeto não é entrar em testes de distribuição normal para a base de dados, outliers e nenhum outro teste estatístico. 
Isso será foco em outro projeto, nesse iremos verificar outros pontos.

# Naive Bayes

```
naive = GaussianNB()
naive.fit(x_treino, y_treino)
```

Métricas do algoritmo

```
accuracy_score(y_treino, previsoes_treino)
print('Acurácia: %.2f%%' % (accuracy_score(y_treino, previsoes_treino)*100))
```

O resultado do teste é: 94.22%

Classification report:

![image](https://github.com/user-attachments/assets/53e237ea-919b-40d1-aa09-b84a2b31ec15)

Validação Cruzada

```
model = GaussianNB()
results = cross_val_score(model, previsores, alvos, cv = kfold)

print('Acurácia: %.2f%%' % (results.mean() * 100.0))
```

Resultado: 93.82%

# SVM

```
svm = SVC(kernel = 'linear', random_state = 1, C = 2)
svm.fit(x_treino, y_treino)
```

Métricas do algoritmo

```
previsores_treino = svm.predict(x_treino)
accuracy_score(y_treino, previsores_treino)
```
Resultado do teste é: 96.48%

Classification report:

![image](https://github.com/user-attachments/assets/a03063c6-8377-47b2-8bb3-c22c89388bdd)

Validação Cruzada

```
kfold = KFold(n_splits = 30, shuffle = True, random_state = 5)
model = SVC(kernel = 'linear', random_state = 1, C = 2)
results = cross_val_score(model, previsores, alvos, cv = kfold)
print('Acurácia: %.2f%%' % (results.mean() * 100))
```

Resultado: 95.06%

# Regressão Logística

```
lr = LogisticRegression(random_state = 1, max_iter = 600,
                        tol = 0.0001, penalty = 'l2', C = 1, solver = 'lbfgs')
lr.fit(x_treino, y_treino)
```

Métricas do algoritmo

```
previsores_treino = lr.predict(x_treino)
accuracy_score(y_treino, previsores_treino)
```

Resultado do teste é: 95.97

Classification report:

![image](https://github.com/user-attachments/assets/63f47c76-67a1-49bd-a486-46b39dc06971)

Validação Cruzada

```
kfold = KFold(n_splits = 30, shuffle = True, random_state = 5)
model = LogisticRegression(random_state = 1, max_iter = 600,
                        tol = 0.0001, penalty = 'l2', C = 1, solver = 'lbfgs')
result = cross_val_score(model, previsores, alvos, cv = kfold)
print("Acurácia: %.2f%%" % (result.mean() * 100.0))
```

Resultado: 95.06%

# KNN

```
knn = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2, algorithm = 'brute')
knn.fit(x_treino, y_treino)
```

Métricas do algoritmo

```
previsores_treino = knn.predict(x_treino)
accuracy_score(y_treino, previsores_treino)
```

Resultado do teste é: 93.71%

Classification report:

![image](https://github.com/user-attachments/assets/51de31de-f217-453a-a5c2-a1479401b765)

Validação Cruzada

```
kfold = KFold(n_splits = 30, shuffle = True, random_state = 5)
model = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2, algorithm = 'brute')
result = cross_val_score(model, previsores, alvos, cv = kfold)

print("Acurácia média: %.2f%%" % (result.mean() * 100.0))
```

Resultado: 92.96%

# Árvores de decisão

```
tree = DecisionTreeClassifier(criterion = 'log_loss', random_state = 0, max_depth = 5, max_leaf_nodes = 6)
tree.fit(x_treino, y_treino)
```

Métricas do algoritmo

```
prev_treino = tree.predict(x_treino)
accuracy_score(y_treino, prev_treino)
```

Resultado do teste é: 96.48%

Classification report:

![image](https://github.com/user-attachments/assets/76e23e89-8c69-49ed-8718-e8efa1e53d55)

Validação Cruzada

```
kfold = KFold(n_splits = 30, shuffle = True, random_state = 5)
model = DecisionTreeClassifier(criterion = 'log_loss', random_state = 0, max_depth = 5, max_leaf_nodes = 6)
results = cross_val_score(model, previsores, alvos, cv = kfold)

print("Acurácia média: %.2f%%" % (results.mean() * 100.0))
```

Resultado: 92.95%

# Random Forest

```
random = RandomForestClassifier(n_estimators = 120, criterion = 'entropy', random_state = 0, max_depth = 3)
random.fit(x_treino, y_treino)
```

Métricas do algoritmo

```
prev_treino = random.predict(x_treino)
print(accuracy_score(y_treino, prev_treino))
```

Resultado do teste é: 97.73%

Classification report:

![image](https://github.com/user-attachments/assets/1278c7d2-59cd-477e-9375-5afa6bae1176)

Validação Cruzada

```
kfold = KFold(n_splits = 30, shuffle = True, random_state = 5)
model = RandomForestClassifier(n_estimators = 120, criterion = 'entropy', random_state = 0, max_depth = 3)
results = cross_val_score(model, previsores, alvos, cv = kfold)

print("Acurácia Média: %.2f%%" % (results.mean() * 100.0))
```

Resultado: 95.98%

# XGBoost

```
xg = XGBClassifier(max_depth = 2, learning_rate = 0.01, n_estimators = 200, objective = 'binary:logistic', random_state = 3)
xg.fit(x_treino, y_treino)
```

Métricas do algoritmo

```
prev_treino = xg.predict(x_treino)
accuracy_score(y_treino, prev_treino)
```

Resultado do teste é: 98.24%

Classification report:

![image](https://github.com/user-attachments/assets/536e4dc3-faad-4564-96d7-3b4db7307996)

Validação Cruzada

```
kfold = KFold(n_splits = 30, shuffle = True, random_state = 5)
model = XGBClassifier(max_depth = 2, learning_rate = 0.01, n_estimators = 200, objective = 'binary:logistic', random_state = 3)
print("Acurácia Média: %.2f%%" % (cross_val_score(model, previsores, alvos, cv = kfold).mean() * 100))
```

Resultado: 95.07%

# LightGBM

```
dataset = lgb.Dataset(x_treino, label = y_treino)
parametros = {'num_leaves': 140,
              'objective': 'binary',
              'max_depth': 2,
              'learning_rate': .03,
              'max_bin': 200}

lgm = lgb.train(parametros, dataset, num_boost_round = 150)
```

Desempenho do algoritmo em relação ao tempo

```
begin = datetime.now()
lgbm = lgb.train(parametros, dataset)
end = datetime.now()
time = end - begin
```

Resultado: datetime.timedelta(microseconds = 64673)

Métricas do algoritmo

```
for i in range(0, 171):
  if prev_lgbm[i] >= 0.5:
    prev_lgbm[i] = 1
  else:
    prev_lgbm[i] = 0

prev_treino = lgbm.predict(x_treino)

for i in range(0, 398):
  if prev_treino[i] >= 0.5:
    prev_treino[i] = 1
  else:
    prev_treino[i] = 0

print("Acurácia: %.2f%%" % (accuracy_score(y_teste, prev_lgbm) * 100))
```

Resultado do teste é: 97.08%

Classification report:

![image](https://github.com/user-attachments/assets/c64ea13d-a9b6-4922-a6c0-80c4e7bef526)

Validação Cruzada

```
kfold = KFold(n_splits = 30, shuffle = True, random_state = 5)
model = lgb.LGBMClassifier(num_leaves = 140, objective = 'binary',
                           max_depth = 2, learning_rate = .03,
                           max_bin = 200)

result = cross_val_score(model, previsores, alvos, cv = kfold)
print("Acurácia Média: %.2f%%" % (result.mean() * 100))
```

Resultado: 95.95%

# CatBoost

```
catboost = CatBoostClassifier(task_type = 'CPU', iterations = 115, learning_rate = 0.015, depth = 6, random_state = 5,
                              eval_metric = 'Accuracy')
```

Métricas do algoritmo

```
prev_cat = catboost.predict(x_teste2)
print("Acurácia: %.2f%%" % (accuracy_score(y_teste2, prev_cat) * 100))
```

Resultado do teste é: 95.91%

Classification report:

![image](https://github.com/user-attachments/assets/ab48cc2f-df79-4c81-80dd-246f83e106c0)

Validação Cruzada

```
kfold = KFold(n_splits = 30, shuffle = True, random_state = 5)
model = CatBoostClassifier(task_type = 'CPU', iterations = 120, learning_rate = 0.012, depth = 6, random_state = 5,
                              eval_metric = 'Accuracy')

results = cross_val_score(model, previsores, alvos, cv = kfold)
print("Acurácia: %.2f%%" % (results.mean() * 100))
```

Resultado: 96.11%

# Conclusões Finais

Algoritmo com melhor acurácia: XGBoost com 98.24%

Algoritmo com maior pontuação na Validação Cruzada: CatBoost com 96.11%
