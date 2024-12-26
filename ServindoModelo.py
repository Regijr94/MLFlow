import pandas as pd 
import requests
import json

# Carregar dados
credito = pd.read_csv("download/Credit.csv")

# Converter colunas categóricas
for col in credito.columns:
    if credito[col].dtype == 'object':
        credito[col] = credito[col].astype('category').cat.codes

# Preparar dados para API
# O MLflow espera um JSON específico com 'dataframe_split' ou 'dataframe_records'
dados = {
    "dataframe_split": {
        "columns": credito.iloc[0:10, 0:20].columns.tolist(),
        "data": credito.iloc[0:10, 0:20].values.tolist()
    }
}

# Fazer requisição
try:
    previsao = requests.post(
        url='http://localhost:2345/invocations', 
        headers={'Content-Type': 'application/json'},
        data=json.dumps(dados)  # Converter para string JSON
    )
    
    previsao.raise_for_status()
    
    print("Status Code:", previsao.status_code)
    print("Previsões:", previsao.json())
    
except requests.exceptions.RequestException as e:
    print(f"Erro na requisição: {e}")
    print("Resposta completa:", previsao.text if 'previsao' in locals() else "Sem resposta")