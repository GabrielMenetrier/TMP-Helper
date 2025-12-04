from flask import Flask, render_template
from markupsafe import Markup
import altair as alt
from utils.visualizacao import main_visualizacao

app = Flask(__name__)

@app.route('/portfolio')
def portfolio():
    # Chama apenas main_visualizacao
    df_final, df_metricas, tickers = main_visualizacao()


    # Gráfico de retorno (Altair)
    df_final_reset = df_final.reset_index().melt('Date', var_name='Modelo', value_name='Retorno')

    print(df_final_reset.head())
    chart_retorno = alt.Chart(df_final_reset).mark_line().encode(
        x='Date:T',
        y='Retorno:Q',
        color='Modelo:N',
        tooltip=['Modelo', 'Retorno', 'Date']
    ).interactive().properties(width=900, height=400)
    chart_retorno_html = chart_retorno.to_html()

    # Gráfico de alpha (IBOV como base zero)
    df_alpha = df_final.copy()
    for col in df_alpha.columns:
        if col != 'IBOV':
            df_alpha[col] = df_alpha[col] - df_alpha['IBOV']
    df_alpha_reset = df_alpha.reset_index().melt('Date', var_name='Modelo', value_name='Alpha')
    chart_alpha = alt.Chart(df_alpha_reset).mark_line().encode(
        x='Date:T',
        y='Alpha:Q',
        color='Modelo:N',
        tooltip=['Modelo', 'Alpha', 'Date']
    ).interactive().properties(width=900, height=400)
    chart_alpha_html = chart_alpha.to_html()

    # Tabela de métricas
    metricas_html = df_metricas.to_html(classes="table table-striped table-hover", border=0)

    return render_template(
        'portfolio.html',
        modelos=tickers,
        metricas_html=Markup(metricas_html),
        chart_retorno_html=Markup(chart_retorno_html),
        chart_alpha_html=Markup(chart_alpha_html)
    )

if __name__ == '__main__':
    app.run(debug=True)