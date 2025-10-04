import matplotlib
matplotlib.use('Agg')
import os
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from pypfopt import EfficientFrontier, risk_models, expected_returns

def baixar_dados(tickers, start_date='2010-01-01', end_date='2025-01-01'):

    data = yf.download(tickers, start=start_date, end=end_date, progress=False).dropna(axis=1, how="all")


    prices = data['Close']

    prices = prices

    returns = prices.pct_change().dropna()

    #Pra que eu vou usar isso?
    retorno_acumulado = (1 + returns).cumprod()
    
    final_retorno_acumulado = retorno_acumulado.iloc[-1] if not retorno_acumulado.empty else pd.Series(1, index=data.columns)


    return data,returns,final_retorno_acumulado

def features_para_cluster(data,retorno,final_retorno_acumulado,meses):

    features = pd.DataFrame(index=data['Close'].columns)


    periodo_dias = meses * 21

    features['Retorno_Medio'] = retorno.mean() * periodo_dias
    features['Volatilidade'] = retorno.std() * np.sqrt(periodo_dias)

    features['Sharpe'] = np.where(
        features['Volatilidade'] > 0,
        features['Retorno_Medio'] / features['Volatilidade'],
        0
        )

    # Pega o último valor de cada coluna (ativo)
    features['Retorno_Acumulado'] = final_retorno_acumulado

    features['Skewness'] = retorno.skew().fillna(0)

    features['Kurtosis'] = retorno.kurtosis().fillna(0)


    cummax = data.cummax()
    drawdown = (data - cummax) / cummax
    features['Max_Drawdown'] = drawdown.min().fillna(0)


    # Correlação média
    corr_matrix = retorno.corr()
    n = len(corr_matrix)
    if n > 1:
        corr_sum = corr_matrix.values.sum() - n  # Subtrair diagonal
        features['Corr_Media'] = corr_sum / (n * (n - 1))
    else:
        features['Corr_Media'] = 0

    imputer = SimpleImputer(strategy='mean')
    features_scaled = imputer.fit_transform(features)


    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)

    return features, features_scaled

def cluster_cotovelo(features_scaled):
    n_samples = len(features_scaled)
    min_k = 2
    max_k = min(8, max(2, n_samples - 1))

    if max_k < min_k:
        optimal_k = 2
        print(f"⚠️ Poucos dados. Usando k={optimal_k}")
        inertias = []
        silhouette_scores = []
        K = [optimal_k]
    else:
        K = list(range(min_k, max_k + 1))
        inertias = []
        silhouette_scores = []
        
        for k in K:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(features_scaled)
                inertias.append(kmeans.inertia_)
                
                if k < n_samples:
                    score = silhouette_score(features_scaled, kmeans.labels_)
                    silhouette_scores.append(score)
            except:
                inertias.append(float('inf'))
                silhouette_scores.append(0)
        
        # Encontrar cotovelo
        if len(inertias) >= 3:
            # Método simples: escolher k onde a redução de inércia diminui
            reductions = np.diff(inertias)
            if len(reductions) > 0:
                optimal_k = min_k + np.argmin(reductions) + 1
            else:
                optimal_k = 3
        else:
            optimal_k = min(3, max_k)
        
        optimal_k = max(min_k, min(optimal_k, max_k))
    return optimal_k, inertias, silhouette_scores

def clusters(features_scaled, optimal_k):

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)

    return kmeans, clusters

def clusters_df(data, clusters, features, final_retorno):

    df_clusters = pd.DataFrame({
        'Ticker': data['Close'].columns,
        'Cluster': clusters
    })

    df_combinado = pd.concat([
        df_clusters.set_index('Ticker'),
        features,
        pd.Series(final_retorno - 1, name='Retorno_Total', index=data['Close'].columns)
    ], axis=1)

    return df_combinado

def portfolio(returns, data, df_combinado, optimal_k):

    tickers_selecionados = []

    for cluster_id in range(optimal_k):
        cluster_data = df_combinado[df_combinado['Cluster'] == cluster_id]
        # Selecionar apenas linhas com Retorno_Total válido
        cluster_data_valid = cluster_data['Retorno_Acumulado'].dropna()
        if not cluster_data_valid.empty:
            best_ticker = cluster_data_valid.idxmax()
            tickers_selecionados.append(best_ticker)
            print(f"Cluster {cluster_id}: {best_ticker} (Retorno: {cluster_data.loc[best_ticker, 'Retorno_Acumulado']*100:.2f}%)")
        else:
            print(f"Cluster {cluster_id} não tem dados válidos e será ignorado.")



    portfolio_all = returns.mean(axis=1)
    retorno_acumulado_all = (1 + portfolio_all).cumprod()
    

    # Carteira selecionada
    if len(tickers_selecionados) > 0:
        dados_selecionados = data['Close'][tickers_selecionados]
        weights = [1/len(tickers_selecionados)] * len(tickers_selecionados)
        portfolio_selected = dados_selecionados.pct_change().dropna().dot(weights)
        retorno_acumulado_selected = (1 + portfolio_selected).cumprod()
    else:
        portfolio_selected = portfolio_all
        retorno_acumulado_selected = retorno_acumulado_all
    return tickers_selecionados, retorno_acumulado_all, retorno_acumulado_selected

def gerar_graficos(features, clusters, retorno,
                retorno_acumulado_all, retorno_acumulado_selected, 
                df_combinado, silhouette_scores, K, inertias,
                optimal_k, tickers_selecionados,
                pesos_inv_vol, pesos_ret, data_inicio_teste, mercado
                ):
    """
    Gera 9 gráficos diferentes e os salva na pasta 'static'.
    """
    os.makedirs('static', exist_ok=True)  # Garante que a pasta 'static' existe



    # Plot 1 =================================================================
    K_valido = []
    inertias_valida = []

    for k_val, inertia in zip(K, inertias):
        if np.isfinite(inertia):
            K_valido.append(k_val)
            inertias_valida.append(inertia)

    plt.figure(figsize=(8,6))
    if len(K_valido) > 1:
        plt.plot(K_valido, inertias_valida, 'bo-', linewidth=2)
        plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'K ótimo = {optimal_k}')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Inércia')
        plt.title('Método do Cotovelo')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5,0.5,f'K = {optimal_k}', ha='center', va='center', fontsize=14)
        plt.title('Clusters')

    plt.tight_layout()
    plt.savefig('static/grafico_cotovelo.png')
    plt.close()
    #========================================================================
    
    
    
    # 2. PCA Visualization ==================================================
    plt.figure(figsize=(8, 6))
    if features.shape[0] > 1 and features.shape[1] > 1:
        try:
            pca = PCA(n_components=2)
            features_pca = pca.fit_transform(features)
            scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], 
                                   c=clusters, cmap='viridis', s=100, alpha=0.7)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
            plt.title('Visualização PCA dos Clusters')
            plt.colorbar(scatter)
        except:
            plt.text(0.5, 0.5, 'PCA não disponível', ha='center', va='center')
            plt.title('PCA')
    else:
        plt.text(0.5, 0.5, 'Dados insuficientes', ha='center', va='center')
        plt.title('PCA')
    plt.tight_layout()
    plt.savefig('static/grafico_pca.png')
    plt.close()
    #========================================================================



    # 3. Comparação de Carteiras ============================================

    # Criar figura e eixo

    mercaados = {
        'BR' : '^BVSP',
        'EUA' : '^GSPC',
        'Conjunto ETFs Macro': '^GSPC',
        'ETFs Globais' : '^GSPC'
    }
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1️⃣ Ibovespa
    ibov = yf.download(mercaados[mercado], start=data_inicio_teste, end=pd.Timestamp.today())
    ibov_retorno = (1 + ibov['Close'].pct_change().fillna(0)).cumprod()
    ibov_retorno.plot(ax=ax, label='Ibovespa', linewidth=2, color='green')

    # 2️⃣ Dados dos tickers selecionados
    dados = yf.download(tickers_selecionados, start=data_inicio_teste, end=pd.Timestamp.today())['Close']
    retorno = dados.pct_change().fillna(0)

    # 3️⃣ Carteira Inverse Volatility
    ret_inv_vol = retorno.dot(pd.Series(pesos_inv_vol))
    ret_acum_inv_vol = (1 + ret_inv_vol).cumprod()
    ret_acum_inv_vol.plot(ax=ax, label='Carteira Inv. Vol.', linewidth=2, linestyle='--', color='blue')

    # 5️⃣ Carteira Retorno Histórico
    ret_ret = retorno.dot(pd.Series(pesos_ret))
    ret_acum_ret = (1 + ret_ret).cumprod()
    ret_acum_ret.plot(ax=ax, label='Carteira Ret. Hist.', linewidth=2, linestyle='--', color='orange')

    # 6️⃣ Todos os tickers (média simples)
    retorno_acumulado_all = (1 + retorno.mean(axis=1)).cumprod()
    retorno_acumulado_all.plot(ax=ax, label='Todos os Tickers', linewidth=2)

    # 7️⃣ Seleção por clusters
    retorno_acumulado_selected = (1 + retorno[tickers_selecionados].mean(axis=1)).cumprod()
    retorno_acumulado_selected.plot(ax=ax, label='Seleção por Clusters', linewidth=2, color='red')

    # Configurações do gráfico
    plt.title('Comparação de Desempenho das Carteiras')
    plt.ylabel('Retorno Acumulado')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/grafico_comparacao_carteiras.png')
    plt.close()

    
    
    
    # 4. Distribuição de Retornos ===========================================
    plt.figure(figsize=(8, 6))
    if not retorno.empty:
        retorno.boxplot(rot=45)
        plt.title('Distribuição de Retornos')
        plt.ylabel('Retorno Diário')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('static/grafico_distribuicao_retornos.png')
    plt.close()
    #========================================================================
    
    
    
    
    # 5. Matriz de Correlação
    plt.figure(figsize=(8, 6))
    if len(tickers_selecionados) > 1 and not retorno.empty:
        corr_selected = retorno[tickers_selecionados].corr()
        sns.heatmap(corr_selected, annot=True, fmt='.2f', cmap='coolwarm', 
                    vmin=-1, vmax=1, cbar_kws={'shrink': 0.8})
        plt.title('Correlação - Carteira Selecionada')
    else:
        plt.text(0.5, 0.5, 'N/A', ha='center', va='center')
        plt.title('Correlação')
    plt.tight_layout()
    plt.savefig('static/grafico_correlacao.png')
    plt.close()

    
    # =======================================================================

    # 6. Risk-Return Map ====================================================
    plt.figure(figsize=(8, 6))
    if not df_combinado.empty:
        for cluster_id in range(optimal_k):
            cluster_data = df_combinado[df_combinado['Cluster'] == cluster_id]
            if len(cluster_data) > 0:
                plt.scatter(cluster_data['Volatilidade']*100, 
                            cluster_data['Retorno_Medio']*100,
                            label=f'Cluster {cluster_id}', s=100, alpha=0.7)
        plt.xlabel('Volatilidade Anual (%)')
        plt.ylabel('Retorno Anual (%)')
        plt.title('Mapa Risco-Retorno')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/grafico_risco_retorno.png')
    plt.close()
    '''
    # 7. Silhouette Scores
    plt.figure(figsize=(8, 6))
    if len(silhouette_scores) > 1:
        plt.plot(K[:len(silhouette_scores)], silhouette_scores, 'go-', linewidth=2)
        plt.axvline(x=optimal_k, color='red', linestyle='--')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'N/A', ha='center', va='center')
        plt.title('Silhouette Score')
    plt.tight_layout()
    plt.savefig('static/grafico_silhouette.png')
    plt.close()
    '''


    '''
    # 8. Performance Metrics
    plt.figure(figsize=(8, 6))
    try:
        metrics = pd.DataFrame({
            'Todos': [
                (retorno_acumulado_all.iloc[-1] - 1) * 100,
                retorno.std() * np.sqrt(252) * 100,
                (retorno.mean() / retorno.std()) * np.sqrt(252) if retorno.std() > 0 else 0
            ],
            'Selecionados': [
                (retorno_acumulado_selected.iloc[-1] - 1) * 100,
                retorno.std() * np.sqrt(252) * 100,
                (retorno.mean() / retorno.std()) * np.sqrt(252) if retorno.std() > 0 else 0
            ]
        }, index=['Retorno Total (%)', 'Volatilidade (%)', 'Sharpe Ratio'])
        
        metrics.plot(kind='bar', alpha=0.8)
        plt.title('Métricas de Performance')
        plt.ylabel('Valor')
        plt.legend(title='Carteira')
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
    except:
        plt.text(0.5, 0.5, 'Métricas não disponíveis', ha='center', va='center')
        plt.title('Métricas')
    plt.tight_layout()
    plt.savefig('static/grafico_metricas.png')
    plt.close()
    '''
    '''
    # 9. Composição da Carteira
    plt.figure(figsize=(8, 6))
    if len(tickers_selecionados) > 0:
        sizes = [100/len(tickers_selecionados)] * len(tickers_selecionados)
        colors = plt.cm.Set3(range(len(tickers_selecionados)))
        plt.pie(sizes, labels=tickers_selecionados, colors=colors, autopct='%1.1f%%')
        plt.title('Composição da Carteira Otimizada')
    else:
        plt.text(0.5, 0.5, 'Sem seleção', ha='center', va='center')
        plt.title('Composição')
    plt.tight_layout()
    plt.savefig('static/grafico_composicao_carteira.png')
    plt.close()
    '''

def mercados(resposta):
    mercados = {
        'BR': [
            # Financeiro
            "ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "SANB11.SA", "BPAC11.SA",
            # Commodities / Energia
            "PETR4.SA", "VALE3.SA", "PRIO3.SA", "CSNA3.SA", "GGBR4.SA",
            "USIM5.SA", "SUZB3.SA", "KLBN11.SA", "BRKM5.SA",
            # Consumo
            "ABEV3.SA", "LREN3.SA", "MGLU3.SA", "VVAR3.SA", "ARZZ3.SA",
            "PCAR3.SA", "NTCO3.SA",
            # Infraestrutura e serviços
            "EQTL3.SA", "ELET3.SA", "ENGI11.SA", "CMIG4.SA", "TAEE11.SA",
            # Saúde
            "HAPV3.SA", "RADL3.SA", "FLRY3.SA",
            # Outros setores
            "RENT3.SA", "B3SA3.SA", "WEGE3.SA", "MULT3.SA", "CYRE3.SA",
        ],
        # 🇺🇸 Mercado Americano — Blue chips, techs e setores tradicionais
        'EUA': [
            # Tech e Comunicação
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX", "ADBE",
            # Financeiro
            "JPM", "GS", "BAC", "MS", "V", "MA", "BLK", "C",
            # Saúde
            "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "CVS",
            # Energia e Indústria
            "XOM", "CVX", "CAT", "BA", "GE", "MMM", "HON",
            # Consumo
            "PG", "KO", "PEP", "MCD", "DIS", "NKE", "COST", "WMT",
            # Diversificação
            "BRK-B", "INTC", "IBM", "ORCL", "AVGO"
        ],
        'ETFs Globais': [
                "SPY",     # S&P 500 (SPDR)
                "IVV",     # S&P 500 (iShares)
                "VOO",     # S&P 500 (Vanguard)
                "VTI",     # Total US Stock Market
                "VT",      # Global Total Market
                "VEA",     # Developed Markets (ex-US)
                "VWO",     # Emerging Markets
            ],

        'Conjunto ETFs Macro': [
                "XLK",     # Tecnologia
                "XLF",     # Financeiro
                "XLV",     # Saúde
                "XLE",     # Energia
                "XLY",     # Consumo Discricionário
                "XLP",     # Consumo Básico
                "XLI",     # Indústria
                "XLRE",    # Real Estate (Imobiliário)
                "XLC",     # Comunicação
                "SMH",     # Semicondutores
                "ARKK",    # Inovação disruptiva (Cathie Wood)
                "QQQ",     # Nasdaq-100
                "SKYY",    # Cloud computing
                "BOTZ",    # Robótica e IA
                "FINX",    # Fintechs
                "CLOU",    # Computação em nuvem
                "IEUR",    # Europa
                "EWG",     # Alemanha
                "EWU",     # Reino Unido
                "EWJ",     # Japão
                "EWW",     # México
                "EWH",     # Hong Kong
                "EWA",     # Austrália
                "EWC",     # Canadá
                "GLD",     # Ouro
                "SLV",     # Prata
                "USO",     # Petróleo WTI
                "DBA",     # Agricultura
                "UUP",     # Dólar americano
                "FXE",     # Euro
                "FXY",     # Iene
                "VNQ",     # Vanguard Real Estate
                "SCHH",    # Schwab US REIT
                "IYR",     # iShares US Real Estate
                "XLRE",    # Real Estate (SPDR)
            ]
        }
    return mercados[resposta]

def separar_datas(meses):
    hoje = pd.Timestamp.today()
    data_passada = hoje - pd.DateOffset(months=meses)
    return data_passada

def marko(tickers, retorno):
    retorno = retorno.replace([np.inf, -np.inf], np.nan).fillna(0)
    mu = expected_returns.mean_historical_return(retorno[tickers])
    S = risk_models.sample_cov(retorno[tickers])

    # 3️⃣ Otimização para máximo Sharpe Ratio
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    return cleaned_weights

def main_otimizacao(mercado,meses):

    tickers = mercados(mercado)

    inicio = separar_datas(meses)

    data, returns, final = baixar_dados(tickers,end_date=inicio)

    features, features_scaled = features_para_cluster(data, returns, final, meses)
    
    optimal_k, inertias, silhouette_scores= cluster_cotovelo(features_scaled)
    
    k_means, cluster = clusters(features_scaled, optimal_k)

    df_combinado = clusters_df(data,cluster,features,final)

    tickers_selecionados, retorno_acumulado_all, retorno_acumulado_selected = portfolio(returns, data, df_combinado, optimal_k)



    ret_med = returns[tickers_selecionados].mean()
    pesos_ret = ret_med / ret_med.sum()

    vols = returns[tickers_selecionados].std()
    pesos_inv_vol = (1/vols) / (1/vols).sum()

    gerar_graficos(features, cluster, returns, retorno_acumulado_all,
    retorno_acumulado_selected, df_combinado, silhouette_scores, list(range(2, 2 + len(inertias))), inertias,
    optimal_k, tickers_selecionados, pesos_inv_vol, pesos_ret, inicio, mercado)

    return pesos_inv_vol, pesos_ret, tickers_selecionados