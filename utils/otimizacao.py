"""
Módulo de otimização: calcula a proporção ótima dos ativos no portfólio.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

def baixar_dados(tickers, start_date='2022-01-01', end_date='2024-01-01'):
    """
    Baixa preços ajustados dos ativos da B3 usando yfinance.
    """
    prices = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
    prices = prices.dropna()
    return prices

def calcular_metricas_anuais(returns, risk_free=0.15):
    """
    Calcula retorno, volatilidade e Sharpe anualizados.
    """
    ret_anual = (1 + returns.mean())**252 - 1
    vol_anual = returns.std() * np.sqrt(252)
    sharpe = (ret_anual - risk_free) / vol_anual
    return ret_anual, vol_anual, sharpe

def portfolio_metrics(weights, expected_returns, cov_matrix, risk_free):
    """
    Calcula retorno, risco e Sharpe do portfólio.
    """
    weights = np.array(weights)
    ret = np.sum(weights * expected_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (ret - risk_free) / vol
    return ret, vol, sharpe

def calcular_portfolio_otimo(tickers, risk_free=0.15):
    """
    Calcula a composição ótima do portfólio (máximo Sharpe, sem vendas a descoberto).
    Retorna proporção de cada ativo e métricas do portfólio.
    """
    # 1. Baixar dados
    prices = baixar_dados(tickers)
    returns = np.log(prices / prices.shift(1)).dropna()
    cov_anual = returns.cov() * 252
    expected_returns = (1 + returns.mean())**252 - 1

    n = len(tickers)

    # 2. Otimização para máximo Sharpe
    def objetivo(weights):
        _, _, sharpe = portfolio_metrics(weights, expected_returns, cov_anual, risk_free)
        return -sharpe

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    x0 = np.ones(n) / n

    result = minimize(objetivo, x0=x0, method='SLSQP', bounds=bounds, constraints=constraints)
    weights_otimos = result.x / result.x.sum()

    # 3. Métricas finais
    ret, vol, sharpe = portfolio_metrics(weights_otimos, expected_returns, cov_anual, risk_free)

    proporcoes = {tickers[i]: round(100*weights_otimos[i], 2) for i in range(n)}
    metricas = {
        'Retorno (%)': f"{ret*100:.2f}",
        'Risco (%)': f"{vol*100:.2f}",
        'Sharpe': f"{sharpe:.3f}"
    }
    return proporcoes, metricas