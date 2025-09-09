from flask import Flask, render_template, request, redirect, url_for, session
from utils.decisao import escolher_ativos
from utils.otimizacao import calcular_portfolio_otimo
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Necessário para usar session

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/formulario')
def formulario():
    return render_template('formulario.html')

@app.route('/processar', methods=['POST'])
def processar():
    respostas = request.form.to_dict()
    # 1. Seleciona ativos com base nas respostas
    ativos = escolher_ativos(respostas)
    # 2. Calcula proporções ótimas do portfólio
    proporcoes, metricas = calcular_portfolio_otimo(ativos)
    # 3. Salva para exibir no resultado
    session['ativos'] = ativos
    session['proporcoes'] = proporcoes
    session['metricas'] = metricas
    return redirect(url_for('resultado'))

@app.route('/resultado')
def resultado():
    ativos = session.get('ativos', [])
    proporcoes = session.get('proporcoes', {})
    metricas = session.get('metricas', {})
    return render_template('resultado.html', ativos=ativos, proporcoes=proporcoes, metricas=metricas)

if __name__ == '__main__':
    app.run(debug=True) #