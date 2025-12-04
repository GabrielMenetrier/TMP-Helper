import pandas as pd
import yfinance as yf
import numpy as np
import altair as alt

def ler_df_caminho(caminho):
    df = pd.read_csv(caminho)
    df["Data"] = pd.to_datetime(df["Data"])
    cols_info = ["Mes", "Data", "Modelo"]
    tickers = [c for c in df.columns if c not in cols_info]
    datas = df["Data"].sort_values().tolist()


    inicio = datas[0]
    fim = datas[-1] + pd.Timedelta(days=40)

        # 2. Baixar preços das ações
    precos = yf.download(tickers, start=inicio, end=fim)["Close"]
    
    if isinstance(precos, pd.Series):
        precos = precos.to_frame()
    
    # 4. Expandir pesos ao longo do tempo
    pesos_diarios = pd.DataFrame(0.0, index=precos.index, columns=tickers)
    for i, data_reb in enumerate(datas):
        linha = df[df["Data"] == data_reb]
        if linha.empty:
            raise ValueError(f"Data de rebalanceamento {data_reb.date()} não existe no CSV")
        pesos = linha[tickers].iloc[0].fillna(0)  # Preencher NaN com 0
        
        # Próximo rebalanceamento ou final
        if i < len(datas)-1:
            prox_data = datas[i+1]
        else:
            prox_data = precos.index[-1]
        
        mask = (pesos_diarios.index >= data_reb) & (pesos_diarios.index < prox_data)
        pesos_diarios.loc[mask, :] = pesos.values

    # ✅ CORREÇÃO: Calcular retornos percentuais
    retornos = precos.pct_change()

    # ✅ Calcular retorno diário do portfólio (soma ponderada)
    portfolio_returns = (retornos * pesos_diarios).sum(axis=1)

    # ✅ Calcular valor acumulado começando em 100
    portfolio_close = (1 + portfolio_returns).cumprod() * 100
    portfolio_close.iloc[0] = 100  # Garantir que começa em 100

    return portfolio_close, inicio, fim, tickers

def ler_df_portifolio(caminhos=['portfolios_mensais_RF_2024-10-01_2025-10-01.csv','portfolios_mensais_media3m_2024-10-01_2025-10-01.csv']):
    dados = {}
    tickers = {}
    for caminho in caminhos:
        portfolio_close, inicio, fim, ticker = ler_df_caminho(caminho)
        dados[caminho.split(sep='_')[2]] = portfolio_close
        tickers[caminho.split(sep='_')[2]] = ticker

    
    # 3. Baixar IBOV
    ibov = yf.download("^BVSP", start=inicio, end=fim)["Close"]

    # ✅ Calcular IBOV normalizado
    ibov_returns = ibov.pct_change()
    ibov_close = (1 + ibov_returns).cumprod() * 100
    ibov_close.iloc[0] = 100

    
    dados['IBOV'] = ibov_close['^BVSP']
    # 6. Juntar tudo
    df_final = pd.DataFrame(dados).dropna()

    return df_final,dados,tickers

def compute_drawdown(series):
    cummax = series.cummax()
    dd = (series - cummax) / cummax
    max_dd = dd.min()

    # Encontrar datas do pico e vale
    valley = dd.idxmin()
    peak = series.loc[:valley].idxmax()

    return dd, max_dd, peak, valley

def compute_metrics(series, benchmark=None, rf_rate=0):
    ret = series.pct_change().dropna()

    ann_factor = 252

    # Retornos
    cumulative = series.iloc[-1] / series.iloc[0] - 1
    daily_mean = ret.mean()
    annual_return = (1 + daily_mean)**ann_factor - 1

    # Risco
    daily_vol = ret.std()
    annual_vol = daily_vol * np.sqrt(ann_factor)

    # Drawdown
    dd_series, max_dd, peak, valley = compute_drawdown(series)

    # VaR e ES
    var_95 = ret.quantile(0.05)
    var_99 = ret.quantile(0.01)
    es_95 = ret[ret <= var_95].mean()
    es_99 = ret[ret <= var_99].mean()

    # Sharpe e Sortino
    downside = ret[ret < 0].std()
    sharpe = (daily_mean - rf_rate/252) / daily_vol if daily_vol != 0 else np.nan
    sortino = (daily_mean - rf_rate/252) / downside if downside != 0 else np.nan
    calmar = annual_return / abs(max_dd) if max_dd != 0 else np.nan

    # Informação vs benchmark
    if benchmark is not None:
        bench_ret = benchmark.pct_change().dropna()
        aligned = ret.align(bench_ret, join="inner")
        te = (aligned[0] - aligned[1]).std() * np.sqrt(ann_factor)
        ir = (annual_return - (1 + bench_ret.mean())**252 + 1) / te if te != 0 else np.nan
        corr = aligned[0].corr(aligned[1])
        beta = aligned[0].cov(aligned[1]) / aligned[1].var()
    else:
        te = ir = corr = beta = np.nan
    monthly = series.resample("M").agg(["first", "last"])
    sinal_mensal = (monthly["last"] > monthly["first"]).mean()
    return {
        "Retorno acumulado": cumulative,
        "Retorno anualizado": annual_return,
        "Retorno médio diário": daily_mean,

        "Volatilidade diária": daily_vol,
        "Volatilidade anualizada": annual_vol,

        "Max Drawdown": max_dd,
        "DD pico": peak,
        "DD vale": valley,

        "VaR 95%": var_95,
        "ES 95%": es_95,
        "VaR 99%": var_99,
        "ES 99%": es_99,

        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,

        "Tracking Error": te,
        "Information Ratio": ir,
        "Correlação com IBOV": corr,
        "Beta vs IBOV": beta,
        "Acerto de sinal mensal": sinal_mensal
    }

def ler_df_metricas(df_series_base_100):
    results = {}
    df = df_series_base_100
    for col in df.columns:
        if col != "IBOV":
            results[col] = compute_metrics(df[col], benchmark=df["IBOV"])
        else:
            results[col] = compute_metrics(df[col])

    # Transformar em DataFrame organizado
    metrics_df = pd.DataFrame(results)

    return metrics_df

    results = {}
    for col in df.columns:
        if col != "IBOV":
            results[col] = compute_metrics(df[col], benchmark=df["IBOV"])
        else:
            results[col] = compute_metrics(df[col])

    # Transformar em DataFrame organizado
    metrics_df = pd.DataFrame(results)

    return metrics_df

def main_visualizacao(caminhos=['portfolios_mensais_RF_2024-10-01_2025-10-01.csv','portfolios_mensais_media3m_2024-10-01_2025-10-01.csv']):
    df_final, dados, tickers = ler_df_portifolio(caminhos)
    df_metricas = ler_df_metricas(df_final)

    return df_final, df_metricas, tickers

def grafico_retorno(df):
    df_reset = df.reset_index().rename(columns={'index': 'Date'})
    df_melt = df_reset.melt('Date', var_name='Modelo', value_name='Retorno')

    highlight = alt.selection_multi(fields=['Modelo'], bind='legend')

    # Paleta Bloomberg/Goldman Sachs
    paleta_bloomberg = [
        "#E0B44C",  # Gold
        "#2F4B7C",
        "#003F5C",
        "#A5A5A5",
        "#2F9E8F",
        "#FF7C43",
        "#D45087",
    ]

    # limites do eixo y (opcional)
    y_min = df_melt["Retorno"].min() * 1
    y_max = df_melt["Retorno"].max() * 1

    base = (
        alt.Chart(df_melt)
        .mark_line(
            strokeWidth=3,
            interpolate="basis"  # linhas suaves
        )
        .encode(
            x=alt.X(
                "Date:T",
                title="Data",
                axis=alt.Axis(
                    labelAngle=0,
                    titleFontSize=18,
                    labelFontSize=14,
                    grid=True,
                    gridOpacity=0.12,
                    gridColor="#CCCCCC",
                ),
            ),
            y=alt.Y(
                "Retorno:Q",
                title="Retorno",
                scale=alt.Scale(domain=[y_min, y_max]),
                axis=alt.Axis(
                    titleFontSize=18,
                    labelFontSize=14,
                    grid=True,
                    gridOpacity=0.15,
                    gridColor="#CCCCCC",
                ),
            ),
            color=alt.Color(
                "Modelo:N",
                scale=alt.Scale(range=paleta_bloomberg),
                legend=alt.Legend(
                    orient="top",
                    title="Modelos",
                    direction="horizontal",
                    columns=3,
                    titleFontSize=18,
                    labelFontSize=16,
                    symbolSize=200,
                    padding=10,
                ),
            ),
            opacity=alt.condition(highlight, alt.value(1), alt.value(0.1)),
            tooltip=[
                alt.Tooltip("Modelo:N", title="Modelo"),
                alt.Tooltip("Retorno:Q", format=".2f", title="Retorno"),
                alt.Tooltip("Date:T", format="%d/%m/%Y", title="Data"),
            ],
        )
        .add_selection(highlight)
        .properties(
            width=900,
            height=450,
            title=alt.TitleParams(
                text="Retorno dos Modelos x IBOV",
                anchor="middle",
                fontSize=26,
                fontWeight="bold",
                font="Helvetica",
            ),
        )
        .configure_view(
            strokeWidth=0,
            fill="#FAFAFA",  # fundo claro estilo bloomberg
        )
        .configure_title(
            color="#333333",
            fontSize=26,
            font="Helvetica Neue",
            anchor="middle",
        )
        .configure_axis(
            domain=False,
            tickColor="#333333",
            labelColor="#333333",
            titleColor="#333333",
        )
        .configure_legend(
            titleColor="#333333",
            labelColor="#333333",
            symbolType="circle",
            symbolStrokeWidth=3,
        )
        .interactive()
    )
    return base

def grafico_alpha(df):
    # ================================
    # PREPARAÇÃO DO DF
    # ================================
    df_reset = df.reset_index().rename(columns={'index': 'Date'})
    
    # Construir dataframe para guardar alphas
    df_alpha = pd.DataFrame()
    df_alpha["Date"] = df_reset["Date"]

    # Calcular alpha = modelo - IBOV
    for col in df_reset.columns:
        if col not in ["Date", "IBOV"]:
            df_alpha[f"Alpha_{col}"] = df_reset[col] - df_reset["IBOV"]

    # Melt para formato longo
    df_alpha_melt = df_alpha.melt(
        "Date", 
        var_name="Modelo", 
        value_name="Alpha"
    )

    # Domínio do eixo Y
    y_min = df_alpha_melt["Alpha"].min() * 1.05
    y_max = df_alpha_melt["Alpha"].max() * 1.05

    # ================================
    # FAIXAS DE FUNDO POSITIVO / NEGATIVO
    # ================================
    fundo_positivo = alt.Chart(pd.DataFrame({
        "y1": [0],
        "y2": [y_max],
        "x1": [df_alpha_melt["Date"].min()],
        "x2": [df_alpha_melt["Date"].max()],
    })).mark_rect(
        color="#D6EDFF",
        opacity=0.45
    ).encode(
        x="x1:T", x2="x2:T", y="y1:Q", y2="y2:Q"
    )

    fundo_negativo = alt.Chart(pd.DataFrame({
        "y1": [y_min],
        "y2": [0],
        "x1": [df_alpha_melt["Date"].min()],
        "x2": [df_alpha_melt["Date"].max()],
    })).mark_rect(
        color="#FFE0E0",
        opacity=0.45
    ).encode(
        x="x1:T", x2="x2:T", y="y1:Q", y2="y2:Q"
    )

    # ================================
    # HIGHLIGHT DE LINHAS (INTERAÇÃO)
    # ================================
    highlight = alt.selection(type="single", on="mouseover", fields=["Modelo"], nearest=True)

    # Paleta Bloomberg (caso não esteja definida)
    paleta_bloomberg = [
        "#3E7CB1", "#F2A900", "#D64C4C", "#1D8348",
        "#7D3C98", "#CA6F1E", "#117864", "#A04000"
    ]

    # ================================
    # GRÁFICO DE LINHAS
    # ================================
    chart_alpha_lines = (
        alt.Chart(df_alpha_melt)
        .mark_line(strokeWidth=3, interpolate="basis")
        .encode(
            x=alt.X(
                "Date:T",
                title="Data",
                axis=alt.Axis(
                    labelAngle=0, labelFontSize=14, titleFontSize=18,
                    grid=True, gridOpacity=0.12, gridColor="#CCCCCC",
                ),
            ),
            y=alt.Y(
                "Alpha:Q",
                title="Alpha (Excesso sobre IBOV)",
                scale=alt.Scale(domain=[y_min, y_max]),
                axis=alt.Axis(
                    labelFontSize=14, titleFontSize=18,
                    grid=True, gridOpacity=0.12, gridColor="#CCCCCC",
                ),
            ),
            color=alt.Color(
                "Modelo:N",
                scale=alt.Scale(range=paleta_bloomberg),
                legend=alt.Legend(
                    orient="top", title="Modelos",
                    columns=3, direction="horizontal",
                    labelFontSize=16, titleFontSize=18,
                    symbolSize=200, padding=10,
                ),
            ),
            opacity=alt.condition(highlight, alt.value(1), alt.value(0.12)),
            tooltip=[
                alt.Tooltip("Modelo:N"),
                alt.Tooltip("Alpha:Q", title="Alpha", format=".4f"),
                alt.Tooltip("Date:T", title="Data", format="%d/%m/%Y"),
            ],
        )
        .add_selection(highlight)
    )

    # ================================
    # COMBINAR FUNDO + LINHAS
    # ================================
    chart_alpha = (
        (fundo_negativo + fundo_positivo + chart_alpha_lines)
        .properties(
            width=900,
            height=450,
            title=alt.TitleParams(
                text="Alpha dos Modelos",
                anchor="middle",
                fontSize=26,
                fontWeight="bold",
                font="Helvetica",
                color="#333",
            ),
        )
        .configure_view(strokeWidth=0, fill="#F8F9FA")
        .configure_axis(domain=False, tickColor="#333", labelColor="#333", titleColor="#333")
        .configure_legend(titleColor="#333", labelColor="#333", symbolType="circle", symbolStrokeWidth=3)
        .interactive()
    )

    return chart_alpha
