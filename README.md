# O objetivo desse projeto prático é utilizar vários modelos de classificação para detectar se uma pessoa tem câncer ou não.

*Tecnologias e bibliotecas utilizadas*: Python, Pandas, NumPy, Matplotlib, SKLEARN, XGBoost, LightGBM e CatBoost.

*Modelos utilizados*: Naive Bayes, SVM, Regressão Logística, KNN, Árvores de Decisão, Random Forest, XGBoost, LighGBM e CatBoost.

## Base de dados inicial

![image](https://github.com/user-attachments/assets/8cf01d5c-d60d-4764-8620-fe553f9ea739)


## Tratamento dos dados

Primeiro tratamento na base de dados é para remover tudo que não for númerico, para fazer isso usamos essa solução:

```df3.replace({'[^0-9]': ''}, regex = True)```

Depois uma função para ajustar os formatos das variáveis númericas, para todas ficarem no mesmo padrão:

```def ajustar_formato(x):

    if isinstance(x, str) and '.' not in x:

        return f'{int(x) / 100:.2f}'

    else:

        return x
```
