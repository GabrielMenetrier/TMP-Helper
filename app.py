from flask import Flask, render_template, request, redirect, url_for
from utils.otimizacao import main_otimizacao, mercados
import os

app = Flask(__name__)
app.secret_key = os.urandom(24) 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/formulario')
def formulario():
    return render_template('formulario.html')

@app.route('/processar', methods=['POST'])
def processar():
    respostas = request.form.to_dict()

    # 1. Seleciona o mercado e o período
    mercado = respostas.get('ativos', 'BR')  # Exemplo: 'BR', 'EUA', 'ETFs'
    meses = int(respostas.get('prazo', 6))  # Período em meses
    proporcao = respostas.get('prop','prop_iguais')

    # 2. Realiza a otimização do portfólio
    pesos_inv_vol, pesos_ret, tickers_selecionados = main_otimizacao(mercado, meses)
    if proporcao == 'inv_var':
        pesos = pesos_inv_vol
        descricao = "Inverso da variância (menor risco)"
    elif proporcao == 'prop_esp':
        pesos = pesos_ret
        descricao = "Retorno esperado (maior risco)"
    else:
        # Proporções iguais
        n = len(tickers_selecionados)
        pesos = {t: 1/n for t in tickers_selecionados}
        descricao = "Proporções iguais (risco balanceado)"



    # 3. Redireciona para a página de resultados com os dados
    return render_template(
        'resultado.html',
        mercado=mercado,
        meses=meses,
        pesos=pesos,
        tickers_selecionados=tickers_selecionados
    )

@app.route('/detalhes')
def detalhes():
    # Lista de gráficos gerados
    graficos = [
        'grafico_cotovelo.png',
        'grafico_pca.png',
        'grafico_comparacao_carteiras.png',
        'grafico_distribuicao_retornos.png',
        'grafico_correlacao.png',
        'grafico_risco_retorno.png',
        'grafico_silhouette.png',
        'grafico_metricas.png',
        'grafico_composicao_carteira.png'
    ]

    return render_template('detalhes.html', graficos=graficos)

if __name__ == '__main__':
    app.run(debug=True)