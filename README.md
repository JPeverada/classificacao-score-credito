# Credit Score Classifier 💳📊

Este projeto consiste em um script Python que utiliza inteligência artificial para classificar o score de crédito de clientes. O modelo analisa dados históricos de perfis financeiros para aprender padrões e, em seguida, é capaz de prever a categoria de score de novos clientes que entram na base.

---

## 🚀 Tecnologias Utilizadas

* **Python 3**
* **Pandas**: Para leitura, manipulação e estruturação das bases de dados (CSV).
* **Scikit-Learn**: Para a construção do pipeline de Machine Learning:
  * `RandomForestClassifier` (Algoritmo de classificação baseado em árvores de decisão).
  * `LabelEncoder` (Para conversão de variáveis categóricas de texto em números).
  * `train_test_split` (Para divisão justa da base entre treino e teste).
  * `accuracy_score` (Métrica para validação da assertividade do modelo).

---

## 🧠 Como o Modelo Funciona

1. **Tratamento de Dados:** As colunas categóricas (`profissao`, `mix_credito` e `comportamento_pagamento`) são transformadas em dados numéricos via `LabelEncoder`.
2. **Divisão da Base:** A base principal (`clientes.csv`) é separada em variáveis preditoras ($X$) e a variável alvo ($y$, que é o `score_credito`). Removem-se IDs ou colunas que não geram valor preditivo.
3. **Treino e Teste:** Os dados são divididos em 70% para treino e 30% para teste, garantindo que a acurácia impressa reflita o desempenho real do modelo em dados não vistos.
4. **Previsão de Novos Clientes:** Após validado, o modelo recebe uma nova base (`novos_clientes.csv`), processa os dados sob a mesma lógica e gera a classificação do score para cada um deles.

---

## 📋 Pré-requisitos e Como Executar

Certifique-se de ter as bibliotecas necessárias instaladas no seu ambiente:

```bash
pip install pandas scikit-learn
