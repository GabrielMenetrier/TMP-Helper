"""
Módulo de decisão: define a regra para selecionar ativos com base nas respostas do usuário.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # Define o backend não interativo
import matplotlib.pyplot as plt
import os

def baixar_dados(tickers, start_date='2022-01-01', end_date='2024-01-01'):
    """
    Baixa preços ajustados dos ativos da B3 usando yfinance.
    """
    # Garantir que todos os tickers sejam strings
    tickers = [str(ticker) for ticker in tickers]

    # Baixar os preços ajustados
    prices = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
    prices = prices.dropna()

    return prices

def calcular_matriz_correlacao(returns):
    """
    Calcula a matriz de correlação entre os retornos diários dos ativos.
    """
    return returns.corr()

def determinar_n_clusters(metricas, max_clusters=10):
    """
    Usa a regra do cotovelo para determinar o número ideal de clusters.
    """
    inertias = []
    for n in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(metricas)
        inertias.append(kmeans.inertia_)

    # Plotar o gráfico da regra do cotovelo
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters + 1), inertias, marker='o', linestyle='--')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inércia')
    plt.title('Regra do Cotovelo para Determinar o Número de Clusters')
    plt.grid(True)

    # Salvar o gráfico na pasta static/
    os.makedirs('static', exist_ok=True)
    plt.savefig('static/elbow_method.png')
    plt.close()

    # Determinar o "cotovelo" (número ideal de clusters)
    # Aqui usamos uma abordagem simples: o ponto onde a redução na inércia começa a diminuir
    deltas = np.diff(inertias)
    cotovelo = np.argmin(deltas / inertias[:-1] < 0.1) + 2  # +2 porque np.diff reduz o índice em 1
    return cotovelo

def clusterizar_ativos_por_correlacao(correlacoes, n_clusters):
    """
    Aplica K-Means para agrupar os ativos com base na matriz de correlação.
    Gera gráficos da clusterização e salva na pasta static/.
    """
    # Converter a matriz de correlação em um DataFrame para o KMeans
    distancias = 1 - correlacoes  # Usamos 1 - correlação como medida de distância
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(distancias)

    # Adicionar os clusters ao DataFrame de correlações
    correlacoes['Cluster'] = clusters

    # Gerar gráfico da clusterização
    plt.figure(figsize=(10, 6))
    for cluster in range(n_clusters):
        cluster_data = correlacoes[correlacoes['Cluster'] == cluster]
        plt.scatter(cluster_data.index, cluster_data.iloc[:, 0], label=f'Cluster {cluster}')
    
    plt.xlabel('Ativos')
    plt.ylabel('Correlação')
    plt.title('Clusterização dos Ativos com Base na Correlação')
    plt.legend()
    plt.grid(True)

    # Salvar o gráfico na pasta static/
    plt.savefig('static/clusterizacao_correlacao.png')
    plt.close()

    return correlacoes

def escolher_ativos(respostas):
    """
    Recebe as respostas do formulário e retorna uma lista de tickers da B3.
    Utiliza clusterização para selecionar os ativos.
    """
    # Lista de ativos disponíveis
    tickers = [
        "BOVA11.SA", "SMAL11.SA", "IVVB11.SA", "PIBB11.SA", "GLDX11.SA", "ALUG11.SA",
        "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA", "BBAS3.SA",
        "MGLU3.SA", "SPXI11.SA", "NASD11.SA"
    ]

    # Baixar dados de preços
    prices = baixar_dados(tickers)

    # Calcular retornos logarítmicos
    returns = np.log(prices / prices.shift(1)).dropna()

    # Calcular a matriz de correlação
    correlacoes = calcular_matriz_correlacao(returns)

    # Determinar o número ideal de clusters usando a regra do cotovelo
    # n_clusters = determinar_n_clusters(correlacoes, max_clusters=min(len(tickers), 10))
    n_clusters = int(respostas.get('n_ativos', 5))
    # Clusterizar ativos com base na correlação
    correlacoes_clusterizadas = clusterizar_ativos_por_correlacao(correlacoes, n_clusters=n_clusters)

    # Selecionar um ativo representativo de cada cluster
    ativos_selecionados = correlacoes_clusterizadas.groupby('Cluster').apply(
        lambda x: x.sample(1, random_state=42)
    ).index.get_level_values(1).tolist()

    return ativos_selecionados