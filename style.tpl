{%- extends 'full.tpl' -%}
{# ------------------------------------------------------------- #}
{# 1. INJEÇÃO DE CSS PERSONALIZADO E REFERÊNCIAS EXTERNAS #}
{# Sobrescreve a seção de cabeçalho para adicionar links e estilos. #}
{% block header %}
    {{ super() }}

    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">

    <style type="text/css">
        /* ==========================================================
        [TLP CSS] ESTILOS DE LIMPEZA E DASHBOARD
        ========================================================== */
        
        /* 1. LAYOUT PRINCIPAL E DIMENSÕES */
        .container { 
            max-width: 1400px; /* Largura máxima do conteúdo */
            margin: auto; 
            padding: 20px;
        }

        /* 2. OCULTAÇÃO DE CÓDIGOS (INPUTS) */
        /* O --no-input do comando é mais eficiente, mas este CSS é uma garantia visual */
        div.input {
            display: none !important; 
        }
        div.prompt {
            display: none !important;
        }

        /* 3. ESTILOS DE DASHBOARD (CUSTOMIZADOS) */
        
        /* Centraliza o gráfico (iframe) */
        .grafico { 
            text-align: center;
            margin-top: 30px; 
            border: 1px solid #ddd; 
            background: #fff; 
            padding: 10px; 
        }

        /* Estilo das ABAS (TR.GROUP) - Cabeçalho do Setor */
        tr.group {
            background-color: #e0e0e0 !important;
            cursor: pointer;
            font-weight: bold;
            font-size: 1.1em;
        }
        tr.group td {
            padding: 18px !important; /* Altura da aba */
            color: #333;
        }
        
        /* Estilo das LINHAS INTERNAS (Ações) */
        table.dataTable tbody td {
            padding: 10px 8px !important; /* Altura da linha interna */
        }
        
        /* Estilo da COLUNA DE AÇÕES (Ticker) */
        table.dataTable tbody td:nth-child(2) { 
            text-align: left !important; 
            font-weight: 600 !important; 
            font-size: 0.9em !important; 
        }
        
        /* Estilo dos ÍCONES (Checks) */
        table.dataTable tbody td:nth-child(n + 3) {
            color: #008000;      /* Cor verde para os ícones */
            font-size: 1.3em;    /* Tamanho dos ícones */
            font-weight: bold;
        }

        /* Remove a borda da saída para ter um look mais limpo */
        .output_wrapper {
            padding: 0 !important;
            border: none !important;
        }
        
    </style>
{% endblock header %}


{# ------------------------------------------------------------- #}
{# 2. INJEÇÃO DE JAVASCRIPT #}
{# Sobrescreve o final do corpo para adicionar scripts JS. #}
{% block body_end %}
    {{ super() }}
    
    <script src="https://code.jquery.com/jquery-3.7.0.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    
    {% endblock body_end %}