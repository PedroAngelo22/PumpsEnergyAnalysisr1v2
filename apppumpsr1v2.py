import streamlit as st
import pandas as pd
import math
import time
import numpy as np
from scipy.optimize import root
import graphviz
import matplotlib.pyplot as plt
import io

# Configura o Matplotlib para não usar um backend de GUI
plt.style.use('seaborn-v0_8-whitegrid')

# --- BIBLIOTECAS DE DADOS (K_FACTORS, FLUIDOS) ---
K_FACTORS = {
    "Entrada de Borda Viva": 0.5, "Entrada Levemente Arredondada": 0.2, "Entrada Bem Arredondada": 0.04,
    "Saída de Tubulação": 1.0, "Válvula Gaveta (Totalmente Aberta)": 0.2, "Válvula Gaveta (1/2 Aberta)": 5.6,
    "Válvula Globo (Totalmente Aberta)": 10.0, "Válvula de Retenção (Tipo Portinhola)": 2.5,
    "Cotovelo 90° (Raio Longo)": 0.6, "Cotovelo 90° (Raio Curto)": 0.9, "Cotovelo 45°": 0.4,
    "Curva de Retorno 180°": 2.2, "Tê (Fluxo Direto)": 0.6, "Tê (Fluxo Lateral)": 1.8,
}
FLUIDOS = { "Água a 20°C": {"rho": 998.2, "nu": 1.004e-6}, "Etanol a 20°C": {"rho": 789.0, "nu": 1.51e-6} }

# --- Funções de Callback ---
def adicionar_item(tipo_lista):
    novo_id = time.time()
    st.session_state[tipo_lista].append({"id": novo_id, "comprimento": 10.0, "diametro": 100.0, "rugosidade": 0.15, "acessorios": []})

def remover_ultimo_item(tipo_lista):
    if len(st.session_state[tipo_lista]) > 0: st.session_state[tipo_lista].pop()

def adicionar_ramal_paralelo():
    novo_nome_ramal = f"Ramal {len(st.session_state.ramais_paralelos) + 1}"
    novo_id = time.time()
    st.session_state.ramais_paralelos[novo_nome_ramal] = [{"id": novo_id, "comprimento": 50.0, "diametro": 80.0, "rugosidade": 0.15, "acessorios": []}]

def remover_ultimo_ramal():
    if len(st.session_state.ramais_paralelos) > 1: st.session_state.ramais_paralelos.popitem()

# --- Funções de Cálculo ---
def calcular_perda_serie(lista_trechos, vazao_m3h, fluido_selecionado):
    perda_total = 0
    for trecho in lista_trechos:
        perdas = calcular_perdas_trecho(trecho, vazao_m3h, fluido_selecionado)
        perda_total += perdas["principal"] + perdas.get("localizada", 0)
    return perda_total

def calcular_perdas_trecho(trecho, vazao_m3h, fluido_selecionado):
    if vazao_m3h <= 0: return {"principal": 0, "localizada": 0, "velocidade": 0}
    vazao_m3s, diametro_m = vazao_m3h / 3600, trecho["diametro"] / 1000
    nu = FLUIDOS[fluido_selecionado]["nu"]
    if diametro_m <= 0: return {"principal": 1e12, "localizada": 0, "velocidade": 0}
    area = (math.pi * diametro_m**2) / 4
    velocidade = vazao_m3s / area
    reynolds = (velocidade * diametro_m) / nu if nu > 0 else 0
    fator_atrito = 0
    if reynolds > 4000:
        rugosidade_m = trecho["rugosidade"] / 1000
        log_term = math.log10((rugosidade_m / (3.7 * diametro_m)) + (5.74 / reynolds**0.9))
        fator_atrito = 0.25 / (log_term**2)
    elif reynolds > 0: fator_atrito = 64 / reynolds
    perda_principal = fator_atrito * (trecho["comprimento"] / diametro_m) * (velocidade**2 / (2 * 9.81))
    return {"principal": perda_principal, "velocidade": velocidade}

def calcular_perdas_paralelo(ramais, vazao_total_m3h, fluido_selecionado):
    num_ramais = len(ramais)
    if num_ramais < 2: return 0, {}
    lista_ramais = list(ramais.values())
    def equacoes_perda(vazoes_parciais_m3h):
        vazao_ultimo_ramal = vazao_total_m3h - sum(vazoes_parciais_m3h)
        if vazao_ultimo_ramal < 0: return [1e12] * (num_ramais - 1)
        todas_vazoes = np.append(vazoes_parciais_m3h, vazao_ultimo_ramal)
        perdas = [calcular_perda_serie(ramal, vazao, fluido_selecionado) for ramal, vazao in zip(lista_ramais, todas_vazoes)]
        erros = [perdas[i] - perdas[-1] for i in range(num_ramais - 1)]
        return erros
    chute_inicial = np.full(num_ramais - 1, vazao_total_m3h / num_ramais)
    solucao = root(equacoes_perda, chute_inicial, method='hybr')
    if not solucao.success: return -1, {}
    vazoes_finais = np.append(solucao.x, vazao_total_m3h - sum(solucao.x))
    perda_final_paralelo = calcular_perda_serie(lista_ramais[0], vazoes_finais[0], fluido_selecionado)
    distribuicao_vazao = {nome_ramal: vazao for nome_ramal, vazao in zip(ramais.keys(), vazoes_finais)}
    return perda_final_paralelo, distribuicao_vazao

def calcular_analise_energetica(vazao_m3h, h_man, eficiencia_bomba, eficiencia_motor, horas_dia, custo_kwh, fluido_selecionado):
    rho = FLUIDOS[fluido_selecionado]["rho"]
    potencia_eletrica_kW = (vazao_m3h / 3600 * rho * 9.81 * h_man) / (eficiencia_bomba * eficiencia_motor) / 1000 if eficiencia_bomba * eficiencia_motor > 0 else 0
    custo_anual = potencia_eletrica_kW * horas_dia * 30 * 12 * custo_kwh
    return {"potencia_eletrica_kW": potencia_eletrica_kW, "custo_anual": custo_anual}

# --- FUNÇÃO PARA O GRÁFICO DE SENSIBILIDADE (REINTEGRADA) ---
def gerar_grafico_sensibilidade_diametro(sistema_base, fator_escala_range, **params_fixos):
    custos = []
    fatores = np.arange(fator_escala_range[0], fator_escala_range[1] + 1, 5) # Passo de 5%

    for fator in fatores:
        escala = fator / 100.0
        sistema_escalado = {
            'antes': [t.copy() for t in sistema_base['antes']],
            'paralelo': {k: [t.copy() for t in v] for k, v in sistema_base['paralelo'].items()},
            'depois': [t.copy() for t in sistema_base['depois']]
        }

        for t in sistema_escalado['antes']: t['diametro'] *= escala
        for _, ramal in sistema_escalado['paralelo'].items():
            for t in ramal: t['diametro'] *= escala
        for t in sistema_escalado['depois']: t['diametro'] *= escala
        
        perda_antes = calcular_perda_serie(sistema_escalado['antes'], params_fixos['vazao'], params_fixos['fluido'])
        perda_par, _ = calcular_perdas_paralelo(sistema_escalado['paralelo'], params_fixos['vazao'], params_fixos['fluido'])
        perda_depois = calcular_perda_serie(sistema_escalado['depois'], params_fixos['vazao'], params_fixos['fluido'])
        
        h_man = params_fixos['h_geo'] + perda_antes + perda_par + perda_depois
        resultado_energia = calcular_analise_energetica(params_fixos['vazao'], h_man, **params_fixos['equipamentos'])
        custos.append(resultado_energia['custo_anual'])

    chart_data = pd.DataFrame({'Fator de Escala nos Diâmetros (%)': fatores, 'Custo Anual de Energia (R$)': custos})
    return chart_data
    
# --- FUNÇÃO PARA GERAR O DIAGRAMA (Existente) ---
def gerar_diagrama_rede(sistema, vazao_total, distribuicao_vazao, fluido):
    dot = graphviz.Digraph(comment='Rede de Tubulação')
    dot.attr('graph', rankdir='LR', splines='ortho')
    dot.attr('node', shape='point')
    dot.node('start', 'Bomba', shape='circle', style='filled', fillcolor='lightblue')
    ultimo_no = 'start'
    
    for i, trecho in enumerate(sistema['antes']):
        proximo_no = f"no_antes_{i+1}"
        velocidade = calcular_perdas_trecho(trecho, vazao_total, fluido)['velocidade']
        label = f"Trecho Antes {i+1}\\n{vazao_total:.1f} m³/h\\n{velocidade:.2f} m/s"
        dot.edge(ultimo_no, proximo_no, label=label)
        ultimo_no = proximo_no
        
    if len(sistema['paralelo']) >= 2:
        no_divisao = ultimo_no; no_juncao = 'no_juncao'; dot.node(no_juncao)
        for nome_ramal, trechos_ramal in sistema['paralelo'].items():
            vazao_ramal = distribuicao_vazao.get(nome_ramal, 0)
            ultimo_no_ramal = no_divisao
            for i, trecho in enumerate(trechos_ramal):
                velocidade = calcular_perdas_trecho(trecho, vazao_ramal, fluido)['velocidade']
                label_ramal = f"{nome_ramal} (T{i+1})\\n{vazao_ramal:.1f} m³/h\\n{velocidade:.2f} m/s"
                if i == len(trechos_ramal) - 1:
                    dot.edge(ultimo_no_ramal, no_juncao, label=label_ramal)
                else:
                    proximo_no_ramal = f"no_{nome_ramal}_{i+1}".replace(" ", "_")
                    dot.edge(ultimo_no_ramal, proximo_no_ramal, label=label_ramal)
                    ultimo_no_ramal = proximo_no_ramal
        ultimo_no = no_juncao
        
    for i, trecho in enumerate(sistema['depois']):
        proximo_no = f"no_depois_{i+1}"
        velocidade = calcular_perdas_trecho(trecho, vazao_total, fluido)['velocidade']
        label = f"Trecho Depois {i+1}\\n{vazao_total:.1f} m³/h\\n{velocidade:.2f} m/s"
        dot.edge(ultimo_no, proximo_no, label=label)
        ultimo_no = proximo_no

    dot.node('end', 'Fim', shape='circle', style='filled', fillcolor='lightgray')
    dot.edge(ultimo_no, 'end')
    return dot

# --- Inicialização do Estado da Sessão ---
if 'trechos_antes' not in st.session_state: st.session_state.trechos_antes = []
if 'trechos_depois' not in st.session_state: st.session_state.trechos_depois = []
if 'ramais_paralelos' not in st.session_state:
    st.session_state.ramais_paralelos = {
        "Ramal 1": [{"id": time.time(), "comprimento": 50.0, "diametro": 80.0, "rugosidade": 0.15}],
        "Ramal 2": [{"id": time.time() + 1, "comprimento": 50.0, "diametro": 100.0, "rugosidade": 0.15}]
    }

# --- Interface do Aplicativo ---
st.set_page_config(layout="wide", page_title="Análise de Redes Hidráulicas")
st.title("💧 Análise de Redes de Bombeamento (Série e Paralelo)")

# --- Barra Lateral ---
with st.sidebar:
    # (A barra lateral continua a mesma)
    st.header("⚙️ Parâmetros Gerais")
    fluido_selecionado = st.selectbox("Selecione o Fluido", list(FLUIDOS.keys()))
    vazao = st.number_input("Vazão Total (m³/h)", 0.1, value=100.0, step=1.0)
    h_geometrica = st.number_input("Altura Geométrica (m)", 0.0, value=15.0)
    st.divider()
    with st.expander("1. Trechos em Série (Antes da Divisão)"):
        for i, trecho in enumerate(st.session_state.trechos_antes):
            with st.container(border=True):
                st.markdown(f"**Trecho {i+1}**"); c1, c2, c3 = st.columns(3)
                trecho['comprimento'] = c1.number_input("L (m)", value=trecho['comprimento'], key=f"comp_antes_{trecho['id']}")
                trecho['diametro'] = c2.number_input("Ø (mm)", value=trecho['diametro'], key=f"diam_antes_{trecho['id']}")
                trecho['rugosidade'] = c3.number_input("Rug. (mm)", value=trecho['rugosidade'], format="%.3f", key=f"rug_antes_{trecho['id']}")
        c1, c2 = st.columns(2); c1.button("Adicionar Trecho (Antes)", on_click=adicionar_item, args=("trechos_antes",), use_container_width=True); c2.button("Remover Trecho (Antes)", on_click=remover_ultimo_item, args=("trechos_antes",), use_container_width=True)
    with st.expander("2. Ramais em Paralelo", expanded=True):
        for nome_ramal, trechos_ramal in st.session_state.ramais_paralelos.items():
            with st.container(border=True):
                st.subheader(f"{nome_ramal}")
                for i, trecho in enumerate(trechos_ramal):
                    st.text(f"Trecho {i+1} do {nome_ramal}"); c1, c2, c3 = st.columns(3)
                    trecho['comprimento'] = c1.number_input("L (m)", value=trecho['comprimento'], key=f"comp_paralelo_{trecho['id']}")
                    trecho['diametro'] = c2.number_input("Ø (mm)", value=trecho['diametro'], key=f"diam_paralelo_{trecho['id']}")
                    trecho['rugosidade'] = c3.number_input("Rug. (mm)", value=trecho['rugosidade'], format="%.3f", key=f"rug_paralelo_{trecho['id']}")
        c1, c2 = st.columns(2); c1.button("Adicionar Ramal Paralelo", on_click=adicionar_ramal_paralelo, use_container_width=True); c2.button("Remover Último Ramal", on_click=remover_ultimo_ramal, use_container_width=True, disabled=len(st.session_state.ramais_paralelos) < 2)
    with st.expander("3. Trechos em Série (Depois da Junção)"):
        for i, trecho in enumerate(st.session_state.trechos_depois):
            with st.container(border=True):
                st.markdown(f"**Trecho {i+1}**"); c1, c2, c3 = st.columns(3)
                trecho['comprimento'] = c1.number_input("L (m)", value=trecho['comprimento'], key=f"comp_depois_{trecho['id']}")
                trecho['diametro'] = c2.number_input("Ø (mm)", value=trecho['diametro'], key=f"diam_depois_{trecho['id']}")
                trecho['rugosidade'] = c3.number_input("Rug. (mm)", value=trecho['rugosidade'], format="%.3f", key=f"rug_depois_{trecho['id']}")
        c1, c2 = st.columns(2); c1.button("Adicionar Trecho (Depois)", on_click=adicionar_item, args=("trechos_depois",), use_container_width=True); c2.button("Remover Trecho (Depois)", on_click=remover_ultimo_item, args=("trechos_depois",), use_container_width=True)
    st.divider()
    st.header("🔌 Equipamentos e Custo")
    rend_bomba = st.slider("Eficiência da Bomba (%)", 1, 100, 70); rend_motor = st.slider("Eficiência do Motor (%)", 1, 100, 90)
    horas_por_dia = st.number_input("Horas por Dia", 1.0, 24.0, 8.0, 0.5); tarifa_energia = st.number_input("Custo da Energia (R$/kWh)", 0.10, 5.00, 0.75, 0.01, format="%.2f")

# --- Lógica Principal e Exibição de Resultados ---
try:
    perda_serie_antes = calcular_perda_serie(st.session_state.trechos_antes, vazao, fluido_selecionado)
    perda_paralelo, distribuicao_vazao = calcular_perdas_paralelo(st.session_state.ramais_paralelos, vazao, fluido_selecionado)
    perda_serie_depois = calcular_perda_serie(st.session_state.trechos_depois, vazao, fluido_selecionado)

    if perda_paralelo == -1:
        st.error("O cálculo do sistema em paralelo falhou. Verifique se os parâmetros dos ramais são consistentes.")
        st.stop()
    
    perda_total_sistema = perda_serie_antes + perda_paralelo + perda_serie_depois
    h_man_total = h_geometrica + perda_total_sistema
    params_equipamentos = {'eficiencia_bomba': rend_bomba/100, 'eficiencia_motor': rend_motor/100, 'horas_dia': horas_por_dia, 'custo_kwh': tarifa_energia, 'fluido_selecionado': fluido_selecionado}
    resultados_energia = calcular_analise_energetica(vazao, h_man_total, **params_equipamentos)

    st.header("📊 Resultados da Análise da Rede (Caso Base)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Altura Manométrica Total", f"{h_man_total:.2f} m"); c2.metric("Perda de Carga Total", f"{perda_total_sistema:.2f} m"); c3.metric("Potência Elétrica", f"{resultados_energia['potencia_eletrica_kW']:.2f} kW"); c4.metric("Custo Anual", f"R$ {resultados_energia['custo_anual']:.2f}")
    
    st.header("🗺️ Diagrama da Rede")
    sistema_atual = {'antes': st.session_state.trechos_antes, 'paralelo': st.session_state.ramais_paralelos, 'depois': st.session_state.trechos_depois}
    diagrama = gerar_diagrama_rede(sistema_atual, vazao, distribuicao_vazao, fluido_selecionado)
    st.graphviz_chart(diagrama)

    st.subheader("Distribuição de Vazão nos Ramais Paralelos")
    if distribuicao_vazao:
        cols = st.columns(len(distribuicao_vazao))
        for i, (nome_ramal, vazao_ramal) in enumerate(distribuicao_vazao.items()):
            trecho_ref = st.session_state.ramais_paralelos[nome_ramal][0]
            velocidade = calcular_perdas_trecho(trecho_ref, vazao_ramal, fluido_selecionado)['velocidade']
            cols[i].metric(f"Vazão no {nome_ramal}", f"{vazao_ramal:.2f} m³/h", f"Velocidade: {velocidade:.2f} m/s")
    
    st.divider()

    # --- SEÇÃO DO GRÁFICO DE SENSIBILIDADE (REINTEGRADA) ---
    with st.expander("📈 Análise de Sensibilidade de Custo por Diâmetro"):
        st.markdown("Veja como o custo anual total do sistema muda ao escalar proporcionalmente todos os diâmetros da sua rede.")
        escala_range = st.slider("Fator de Escala para Diâmetros (%)", min_value=50, max_value=200, value=(80, 120), help="Define a faixa de variação para os diâmetros do seu 'Caso Base'.")
        
        sistema_base = {'antes': st.session_state.trechos_antes, 'paralelo': st.session_state.ramais_paralelos, 'depois': st.session_state.trechos_depois}
        params_fixos = {'vazao': vazao, 'h_geo': h_geometrica, 'fluido': fluido_selecionado, 'equipamentos': params_equipamentos}
        
        chart_data = gerar_grafico_sensibilidade_diametro(sistema_base, escala_range, **params_fixos)
        
        st.line_chart(chart_data.set_index('Fator de Escala nos Diâmetros (%)'))

except Exception as e:
    st.error(f"Ocorreu um erro durante o cálculo. Verifique os parâmetros de entrada. Detalhe: {e}")
