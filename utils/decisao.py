"""
Módulo de decisão: define a regra para selecionar ativos com base nas respostas do usuário.
"""

def escolher_ativos(respostas):
    """
    Recebe as respostas do formulário e retorna uma lista de tickers da B3.
    Por enquanto, retorna uma lista fixa de exemplo.
    """
    print(respostas)
    return ['ITUB4.SA', 'VALE3.SA', 'WEGE3.SA']