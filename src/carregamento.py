

import os
import pandas as pd
import json

# Diretórios base
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DADOS_DIR = os.path.join(BASE_DIR, 'data')

# Subpastas
PASTA_JOGADORES = os.path.join(DADOS_DIR, 'players')
PASTA_EQUIPES = os.path.join(DADOS_DIR, 'teams')
PASTA_GERAL = os.path.join(DADOS_DIR, 'geral')


def carregar_csvs(diretorio: str) -> dict:
    """
    Carrega todos os arquivos CSV de um diretório em um dicionário.
    Retorna: {nome_arquivo_sem_extensão: DataFrame}
    """
    dataframes = {}
    for arquivo in os.listdir(diretorio):
        if arquivo.endswith('.csv'):
            caminho = os.path.join(diretorio, arquivo)
            nome = os.path.splitext(arquivo)[0]
            df = pd.read_csv(caminho)
            dataframes[nome] = df
            print(f"✓ {nome} carregado: {df.shape}")
    return dataframes


def carregar_json(caminho_json: str) -> dict:
    """Carrega um arquivo JSON como dicionário Python."""
    with open(caminho_json, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    print("📂 Carregando dados...\n")

    # Dados dos jogadores
    print("🔹 Jogadores:")
    dados_jogadores = carregar_csvs(PASTA_JOGADORES)
    descricao_jogadores = carregar_json(os.path.join(PASTA_JOGADORES, 'player_data_description.json'))

    # Dados das equipes
    print("\n🔹 Equipes:")
    dados_equipes = carregar_csvs(PASTA_EQUIPES)
    descricao_equipes = carregar_json(os.path.join(PASTA_EQUIPES, 'team_tips.json'))

    # Dados gerais (partidas)
    print("\n🔹 Partidas:")
    dados_geral = carregar_csvs(PASTA_GERAL)

    print("\n✅ Todos os dados foram carregados com sucesso!")
