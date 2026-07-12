import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

tabela = pd.read_csv('Docs/clientes.csv')

cod_profissao = LabelEncoder()
cod_mix = LabelEncoder()
cod_pagamento = LabelEncoder()

tabela["profissao"] = cod_profissao.fit_transform(tabela["profissao"])
tabela["mix_credito"] = cod_mix.fit_transform(tabela["mix_credito"])
tabela["comportamento_pagamento"] = cod_pagamento.fit_transform(tabela["comportamento_pagamento"])

y = tabela["score_credito"]
x = tabela.drop(["score_credito", "id_cliente"], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

modelo = RandomForestClassifier()
modelo.fit(x_train, y_train)

previsoes = modelo.predict(x_test)
print("Acurácia do modelo:", accuracy_score(y_test, previsoes))

tabela_novos_clientes = pd.read_csv('Docs/novos_clientes.csv')

tabela_novos_clientes["profissao"] = cod_profissao.transform(tabela_novos_clientes["profissao"])
tabela_novos_clientes["mix_credito"] = cod_mix.transform(tabela_novos_clientes["mix_credito"])
tabela_novos_clientes["comportamento_pagamento"] = cod_pagamento.transform(tabela_novos_clientes["comportamento_pagamento"])

print(tabela_novos_clientes)

nova_previsao = modelo.predict(tabela_novos_clientes)
print("Previsão para os novos clientes:", nova_previsao)
