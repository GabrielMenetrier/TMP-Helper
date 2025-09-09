"""
Módulo de decisão: define a regra para selecionar ativos com base nas respostas do usuário.
"""

def escolher_ativos(respostas):
    """
    Recebe as respostas do formulário e retorna uma lista de tickers da B3.
    Por enquanto, retorna uma lista fixa de exemplo.
    """
    print(respostas)
    ativos = [
    "HASH11.SA",   # Bitcoin
    "IVVB11.SA",
    "GOLD11.SA",   # Ouro / Commodities
    "BOVA11.SA",   # Índice Bovespa
    "IMAB11.SA",   # Renda Fixa / Tesouro IPCA
    "FIXA11.SA",   # Renda Fixa / CDI
    ]
    
    return ativos