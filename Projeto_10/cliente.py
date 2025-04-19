# Projeto 10 - Construindo API Para Deploy do Modelo de Deep Learning
# Cliente

# Imports
import requests
import json

# URL da API
url = 'http://127.0.0.1:5000/predict'

# Dados a serem enviados como JSON
dados_dsa =[
    {
        "indice_vegetacao": 372,
        "capacidade_solo": 690,
        "concentracao_co2": 3781.5,
        "nivel_nutrientes": 920.45,
        "indice_fertilizantes": 860.32,
        "profundidade_raiz": 419.28,
        "radiacao_solar": 902,
        "precipitacao": 52.13,
        "estagio_crescimento": 159.84712,
        "historico_rendimento": 251.6
    },
    {
        "indice_vegetacao": 348,
        "capacidade_solo": 677,
        "concentracao_co2": 3695.2,
        "nivel_nutrientes": 907.31,
        "indice_fertilizantes": 838.45,
        "profundidade_raiz": 408.91,
        "radiacao_solar": 879,
        "precipitacao": 47.95,
        "estagio_crescimento": 150.31892,
        "historico_rendimento": 240.7
    },
    {
        "indice_vegetacao": 361,
        "capacidade_solo": 688,
        "concentracao_co2": 3748.7,
        "nivel_nutrientes": 915.78,
        "indice_fertilizantes": 853.91,
        "profundidade_raiz": 414.62,
        "radiacao_solar": 895,
        "precipitacao": 51.42,
        "estagio_crescimento": 157.21543,
        "historico_rendimento": 247.9
    },
    {
        "indice_vegetacao": 359,
        "capacidade_solo": 682,
        "concentracao_co2": 3722.9,
        "nivel_nutrientes": 911.85,
        "indice_fertilizantes": 845.67,
        "profundidade_raiz": 411.03,
        "radiacao_solar": 888,
        "precipitacao": 50.23,
        "estagio_crescimento": 155.67481,
        "historico_rendimento": 244.5
    },
    {
        "indice_vegetacao": 366,
        "capacidade_solo": 695,
        "concentracao_co2": 3801.2,
        "nivel_nutrientes": 925.34,
        "indice_fertilizantes": 868.12,
        "profundidade_raiz": 423.56,
        "radiacao_solar": 910,
        "precipitacao": 53.67,
        "estagio_crescimento": 162.48291,
        "historico_rendimento": 256.3
    },
    {
        "indice_vegetacao": 351,
        "capacidade_solo": 678,
        "concentracao_co2": 3708.5,
        "nivel_nutrientes": 909.67,
        "indice_fertilizantes": 841.32,
        "profundidade_raiz": 409.76,
        "radiacao_solar": 881,
        "precipitacao": 48.67,
        "estagio_crescimento": 152.84673,
        "historico_rendimento": 242.9
    },
    {
        "indice_vegetacao": 357,
        "capacidade_solo": 680,
        "concentracao_co2": 3715.9,
        "nivel_nutrientes": 912.12,
        "indice_fertilizantes": 843.89,
        "profundidade_raiz": 410.87,
        "radiacao_solar": 884,
        "precipitacao": 49.36,
        "estagio_crescimento": 154.11234,
        "historico_rendimento": 243.8
    },
    {
        "indice_vegetacao": 370,
        "capacidade_solo": 692,
        "concentracao_co2": 3790.6,
        "nivel_nutrientes": 923.78,
        "indice_fertilizantes": 865.41,
        "profundidade_raiz": 421.02,
        "radiacao_solar": 905,
        "precipitacao": 52.89,
        "estagio_crescimento": 161.32478,
        "historico_rendimento": 254.1
    },
    {
        "indice_vegetacao": 355,
        "capacidade_solo": 685,
        "concentracao_co2": 3730.2,
        "nivel_nutrientes": 913.45,
        "indice_fertilizantes": 847.23,
        "profundidade_raiz": 412.98,
        "radiacao_solar": 892,
        "precipitacao": 50.89,
        "estagio_crescimento": 156.54912,
        "historico_rendimento": 246.7
    },
    {
        "indice_vegetacao": 368,
        "capacidade_solo": 690,
        "concentracao_co2": 3775.3,
        "nivel_nutrientes": 921.56,
        "indice_fertilizantes": 862.78,
        "profundidade_raiz": 418.42,
        "radiacao_solar": 899,
        "precipitacao": 52.56,
        "estagio_crescimento": 160.11832,
        "historico_rendimento": 252.5
    }
]

# Headers específicos para definir o tipo de conteúdo como JSON
headers = {'Content-Type': 'application/json'}

# Faz a requisição POST
response = requests.post(url, headers = headers, data = json.dumps(dados_dsa))

# Verifica se a requisição foi bem sucedida
if response.status_code == 200:
    print("\nResposta da API:", response.json())
    print("\n")
else:
    print("Erro na requisição:", response.status_code, response.text)
