import pandas as pd

df = pd.DataFrame({'Categoria': ["teste", "prod"]})
print(df)

teste = pd.get_dummies(df, columns=['Categoria'])

# Exemplo de DataFrame com uma coluna categórica
df = pd.DataFrame({
    'Categoria': ['A', 'B', 'A', 'C']
})

# Usando get_dummies para converter a coluna categórica em variáveis dummy
df_dummies = pd.get_dummies(df, columns=['Categoria'])

print(df_dummies)