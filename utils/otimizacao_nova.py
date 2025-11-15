import requests
import pandas as pd
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
import yfinance as yf


from datetime import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf

import yfinance as yf
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, WMAIndicator, MACD, CCIIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, VolumeWeightedAveragePrice

import yfinance as yf
from ta.trend import SMAIndicator, EMAIndicator, MACD, CCIIndicator, ADXIndicator, WMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from ta.volatility import AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice

def baixar_e_calcular_indicadores(ticker, start="2010-01-01"):
    data = yf.download(ticker, start=start, progress=False)
    serie = data['Close'][ticker]
    data['SMA_5'] = SMAIndicator(serie, window=5).sma_indicator()
    data['SMA_10'] = SMAIndicator(serie, window=10).sma_indicator()
    data['SMA_20'] = SMAIndicator(serie, window=20).sma_indicator()
    data['SMA_50'] = SMAIndicator(serie, window=50).sma_indicator()
    data['SMA_100'] = SMAIndicator(serie, window=100).sma_indicator()
    data['SMA_200'] = SMAIndicator(serie, window=200).sma_indicator()

    data['EMA_12'] = EMAIndicator(serie, window=12).ema_indicator()
    data['EMA_26'] = EMAIndicator(serie, window=26).ema_indicator()
    data['EMA_50'] = EMAIndicator(serie, window=50).ema_indicator()
    data['EMA_200'] = EMAIndicator(serie, window=200).ema_indicator()

    data['WMA_20'] = WMAIndicator(serie, window=20).wma()
    data['RSI_14'] = RSIIndicator(serie, window=14).rsi()

    macd = MACD(data['Close'][ticker])
    data['MACD'] = macd.macd()
    data['MACD_Hist'] = macd.macd_diff()

    stoch = StochasticOscillator(data['High'][ticker], data['Low'][ticker], data['Close'][ticker])
    data['Stoch_K'] = stoch.stoch()
    data['Stoch_D'] = stoch.stoch_signal()

    data['CCI_20'] = CCIIndicator(data['High'][ticker], data['Low'][ticker], data['Close'][ticker], window=20).cci()
    data['ADX_14'] = ADXIndicator(data['High'][ticker], data['Low'][ticker], data['Close'][ticker], window=14).adx()
    data['ATR_14'] = AverageTrueRange(data['High'][ticker], data['Low'][ticker], data['Close'][ticker], window=14).average_true_range()
    data['ROC_12'] = ROCIndicator(data['Close'][ticker], window=12).roc()

    data['OBV'] = OnBalanceVolumeIndicator(data['Close'][ticker], data['Volume'][ticker]).on_balance_volume()
    data['CMF_20'] = ChaikinMoneyFlowIndicator(
        data['High'][ticker], data['Low'][ticker], data['Close'][ticker], data['Volume'][ticker], window=20
    ).chaikin_money_flow()

    data['VWAP'] = VolumeWeightedAveragePrice(
        data['High'][ticker], data['Low'][ticker], data['Close'][ticker], data['Volume'][ticker]
    ).volume_weighted_average_price()

    # -----------------------------
    # Retornos
    # -----------------------------
    data['Ret_1d'] = data['Close'].pct_change(1)
    data['Ret_5d'] = data['Close'].pct_change(5)
    data['Ret_21d'] = data['Close'].pct_change(21)

    # -----------------------------
    # Volatilidade
    # -----------------------------
    data['Vol_21d'] = data['Close'].rolling(21).std()
    data['Vol_63d'] = data['Close'].rolling(63).std()
    data['Vol_252d'] = data['Close'].rolling(252).std()

    data['Ticker'] = ticker
    
    return data

"""
def baixar_df_fundamentalista(ticker):

    df, df_annualy, df_quarterly = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    Ticker = yf.Ticker(ticker)

    df_quarterly = Ticker.quarterly_financials.loc[['Total Revenue','Operating Income','EBITDA','Net Income','Research And Development']].transpose().dropna()
    df_annualy = Ticker.financials.loc[['Total Revenue','Operating Income','EBITDA','Net Income','Research And Development']].transpose().dropna()

    primeira_observação_quarterly = df_quarterly.index.min()

    df_annualy = df_annualy.loc[df_annualy.index < primeira_observação_quarterly]/4
    df = pd.concat([df_quarterly, df_annualy]).sort_index()

    return df

def baixar_df_recommendations(ticker):


    df = yf.Ticker(ticker).recommendations
    if "period" not in df.columns:
        raise ValueError("O DataFrame precisa conter a coluna 'period'.")

    base_date = datetime.today().replace(day=1)

    def period_to_date(p):
        months = int(p.replace("m", ""))
        # mantém só a parte da data (YYYY-MM-DD)
        return (base_date + relativedelta(months=months)).date()

    df = df.copy()
    df["date"] = df["period"].apply(period_to_date)
    df = df.drop(columns=["period"]).set_index("date").sort_index()
    return df

def baixar_df_returns(ticker, media_movel_curta=10, media_movel_longa=30):
    df = yf.Ticker(ticker).history(period='max')[['Close','Volume']]
    df['Returns'] = df['Close'].pct_change().dropna()
    df["MedMovelCurta"] = df['Close'].rolling(window=media_movel_curta).mean()
    df["MedMovelLonga"] = df['Close'].rolling(window=media_movel_longa).mean()
    df['RSI'] = (df["MedMovelLonga"] > df["MedMovelCurta"]).astype(int)
    df['Vol'] = df['Close'].rolling(window=14).std()
    return df

def baixar_df_macro(start_date='2020-01-01', end_date='2024-12-31'):
    API_KEY = '92fab9d0baa168aac19e43e8866b4153'
    
    def get_data(series_id):
        url = 'https://api.stlouisfed.org/fred/series/observations'
        params = {
            'series_id': series_id,
            'api_key': API_KEY,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date
        }
        r = requests.get(url, params=params)
        
        if r.status_code != 200:
            print(f"Erro HTTP {r.status_code} ao buscar {series_id}")
            print(f"Resposta: {r.text}")
            return None
        
        result = r.json()
        
        if 'observations' not in result:
            print(f"Erro ao buscar {series_id}:")
            print(f"Resposta completa: {result}")
            return None
        
        data = result['observations']
        df_temp = pd.DataFrame(data)
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        df_temp['value'] = pd.to_numeric(df_temp['value'], errors='coerce')
        return df_temp[['date', 'value']].dropna()

    series = {
        'eur_usd': 'DEXUSEU',
        'cpi': 'CPIAUCSL',
        'fed_rate': 'DFF',
        'gdp': 'GDP',
        'unemployment': 'UNRATE',
        'sp500': 'SP500'
    }
    
    df = pd.DataFrame()
    for name, series_id in series.items():
        temp = get_data(series_id)
        if temp is not None:
            temp = temp.rename(columns={'value': name})
            df = temp if df is None else pd.merge(df, temp, on='date', how='outer')

    df = df.sort_values('date').reset_index(drop=True)
    df = df.ffill().set_index('date')  # ffill() ao invés de fillna(method='ffill')
    return df




def baixar_dfs(tickers):
    dfs = {}
    
    for ticker in tickers:
        try:
            # Baixar os três DataFrames
            fundamentalista = baixar_df_fundamentalista(ticker)
            recommendations = baixar_df_recommendations(ticker)
            returns = baixar_df_returns(ticker).tz_localize(None)
            
            # Garantir que todos os índices são datetime
            if not isinstance(fundamentalista.index, pd.DatetimeIndex):
                fundamentalista.index = pd.to_datetime(fundamentalista.index)
            if not isinstance(recommendations.index, pd.DatetimeIndex):
                recommendations.index = pd.to_datetime(recommendations.index)
            if not isinstance(returns.index, pd.DatetimeIndex):
                returns.index = pd.to_datetime(returns.index)
            
            # Criar um índice comum baseado no returns (diário)
            date_range = returns.index
            
            # Reindexar fundamentalista e recommendations para o índice diário
            # usando forward fill para preencher os dias intermediários
            fundamentalista_daily = fundamentalista.reindex(date_range, method='ffill')
            recommendations_daily = recommendations.reindex(date_range, method='ffill')
            
            # Adicionar prefixo ao nome das colunas para identificar a origem
            fundamentalista_daily = fundamentalista_daily.add_prefix('fund_')
            recommendations_daily = recommendations_daily.add_prefix('rec_')
            returns = returns.add_prefix('ret_')
            
            # Combinar os três DataFrames
            df_combined = pd.concat([
                fundamentalista_daily,
                recommendations_daily,
                returns
            ], axis=1)
            columns = df_combined.columns.tolist()
            # Remover linhas com todos os valores NaN
            df_combined = df_combined.dropna(how='all')
            

            for col in columns:

                df_combined[col + "_missing"] = df_combined[col].isna().astype(int)


            # Armazenar no dicionário
            dfs[ticker] = df_combined
            
            print(f"✓ {ticker}: {len(df_combined)} observações")
            
        except Exception as e:
            print(f"✗ Erro ao processar {ticker}: {str(e)}")
            continue
    
    return dfs
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score


def prever_ols(df, target_lag=5):
    """
    df: DataFrame já com indicadores técnicos calculados.
    target_lag: número de dias no futuro que queremos prever o retorno.
                Ex: 5 = prever retorno em 5 dias.

    Retorna:
        - modelo OLS treinado
        - previsões
        - R²
        - RMSE
        - DataFrame com coluna 'Pred_OLS'
    """

    data = df.copy()

    # -----------------------------
    # Criar variável alvo (y)
    # -----------------------------
    data["Target"] = data["Close"].pct_change(target_lag).shift(-target_lag)

    # Remover linhas sem retorno futuro
    data = data.dropna(subset=["Target"]).copy()

    # -----------------------------
    # Selecionar features numéricas
    # -----------------------------
    features = data.select_dtypes(include=[np.number]).drop(
        columns=["Target"], errors="ignore"
    )

    # Remover colunas constantes
    features = features.loc[:, features.std() > 0]

    # y
    y = data["Target"]

    # X com constante
    X = sm.add_constant(features)

    # -----------------------------
    # Ajustar o modelo OLS
    # -----------------------------
    model = sm.OLS(y, X).fit()

    # -----------------------------
    # Previsões
    # -----------------------------
    y_pred = model.predict(X)

    # -----------------------------
    # Métricas
    # -----------------------------
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # -----------------------------
    # Salvar previsões no DF
    # -----------------------------
    data["Pred_OLS"] = y_pred

    return model, y_pred, r2, rmse, data
