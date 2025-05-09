import os
import pandas as pd
import matplotlib.pyplot as plt

# Caminho base
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RELATORIO_DIR = os.path.join(BASE_DIR, 'relatorio', 'graficos')

# Garante que o diretório de gráficos exista
os.makedirs(RELATORIO_DIR, exist_ok=True)

# Caminho do DataFrame unificado da Etapa 2
CAMINHO_JOGADORES = os.path.join(BASE_DIR, 'data', 'df_jogadores.csv')

# Função para salvar gráfico de barras
def salvar_grafico_barra(dados, titulo, xlabel, ylabel, nome_arquivo):
    plt.figure(figsize=(10, 6))
    plt.barh(dados['player'], dados[nome_arquivo], color='steelblue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(titulo)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    caminho = os.path.join(RELATORIO_DIR, f"{nome_arquivo}.png")
    plt.savefig(caminho)
    plt.close()
    print(f"📊 Gráfico salvo: {caminho}")

# Carrega os dados unificados
df = pd.read_csv(CAMINHO_JOGADORES)

# Remove duplicatas e nulos em colunas principais
df = df.drop_duplicates(subset=['player', 'team'])

# Lista de métricas e títulos que vamos gerar
metricas = {
    'goals': 'Top 10 Jogadores com Mais Gols',
    'assists': 'Top 10 Jogadores com Mais Assistências',
    'xg': 'Top 10 em Expected Goals (xG)',
    'xg_assist': 'Top 10 em Expected Assists (xA)',
    'goals_per90': 'Top 10 Gols por 90 minutos',
    'assists_per90': 'Top 10 Assistências por 90 minutos',
    'gca': 'Top 10 em Ações que Geram Gols (GCA)',
    'sca': 'Top 10 em Ações que Geram Chutes (SCA)',
    'passes_completed': 'Top 10 Jogadores com Mais Passes Completos',
    'dribbles_completed': 'Top 10 Jogadores com Mais Dribles Completos',
    'minutes_90s': 'Top 10 Jogadores com Mais Minutos Jogados',
    'cards_yellow': 'Top 10 em Cartões Amarelos',
    'cards_red': 'Top 10 em Cartões Vermelhos',
}

# Geração de gráficos
for metrica, titulo in metricas.items():
    if metrica in df.columns:
        top10 = df[['player', metrica]].dropna().sort_values(by=metrica, ascending=False).head(10)
        salvar_grafico_barra(
            dados=top10,
            titulo=titulo,
            xlabel=metrica.replace('_', ' ').title(),
            ylabel='Jogador',
            nome_arquivo=metrica
        )

# Estatísticas gerais
estatisticas = df.describe(include='all')
estatisticas_path = os.path.join(RELATORIO_DIR, 'estatisticas_descritivas.csv')
estatisticas.to_csv(estatisticas_path)
print(f"📄 Estatísticas descritivas salvas em: {estatisticas_path}")

print("\n✅ Etapa 3 concluída: gráficos e estatísticas gerados com sucesso.")
