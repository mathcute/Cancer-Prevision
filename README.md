# O objetivo desse projeto prático é utilizar vários modelos de classificação para detectar se uma pessoa tem câncer ou não.

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
accuracy_score(y_teste, previsoes)
print('Acurácia: %.2f%%' % (accuracy_score(y_teste, previsoes)*100))
```

O resultado é: 92.40%
Classification report:

![image](https://github.com/user-attachments/assets/53e237ea-919b-40d1-aa09-b84a2b31ec15)
