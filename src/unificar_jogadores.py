import os
import pandas as pd
from functools import reduce

# Caminhos base
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DADOS_JOGADORES = os.path.join(BASE_DIR, 'data', 'players')

def carregar_todos_csvs(diretorio):
    """
    Carrega todos os arquivos .csv de um diretório e retorna um dicionário {nome: DataFrame}
    """
    dataframes = {}
    for arquivo in os.listdir(diretorio):
        if arquivo.endswith('.csv'):
            nome = os.path.splitext(arquivo)[0]
            caminho = os.path.join(diretorio, arquivo)
            df = pd.read_csv(caminho)
            dataframes[nome] = df
            print(f"✓ {nome} carregado: {df.shape}")
    return dataframes

def normalizar_colunas(df):
    """
    Padroniza os nomes das colunas: minúsculas, sem espaços, com underscores.
    """
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

def unificar_dataframes_por_jogador(dfs_dict):
    """
    Junta todos os DataFrames usando 'player' e 'team' como chave,
    resolve conflitos de colunas e mantém apenas uma versão de cada métrica.
    """
    chaves = ['player', 'team']
    dfs_normalizados = []

    for nome, df in dfs_dict.items():
        df = normalizar_colunas(df)
        if all(chave in df.columns for chave in chaves):
            df = df.loc[:, ~df.columns.duplicated()]
            dfs_normalizados.append(df)

    def merge_seguro(left, right):
        return pd.merge(left, right, on=chaves, how='outer', suffixes=('', f'_{right.shape[1]}'))

    df_unificado = reduce(merge_seguro, dfs_normalizados)

    # Remove colunas com sufixos duplicados, mantendo apenas uma versão por nome base
    colunas_unicas = {}
    for col in df_unificado.columns:
        base = col.split('_')[0]
        if base not in colunas_unicas:
            colunas_unicas[base] = col

    df_unificado = df_unificado[list(colunas_unicas.values())]
    return df_unificado

if __name__ == "__main__":
    print("📊 Unificando dados dos jogadores...")

    # Carrega todos os CSVs
    dados_jogadores = carregar_todos_csvs(DADOS_JOGADORES)

    # Unifica os DataFrames por 'player' e 'team'
    df_jogadores = unificar_dataframes_por_jogador(dados_jogadores)

    print(f"\n✅ Base final unificada: {df_jogadores.shape[0]} jogadores, {df_jogadores.shape[1]} atributos\n")
    print("📌 Colunas disponíveis:")
    print(df_jogadores.columns.tolist()[:15], '...')

    print("\n🔍 Amostra:")
    print(df_jogadores[['player', 'team']].head())

    # Salva o DataFrame unificado para uso na etapa seguinte
    CAMINHO_SAIDA = os.path.join(BASE_DIR, 'data', 'df_jogadores.csv')
    df_jogadores.to_csv(CAMINHO_SAIDA, index=False)
    print(f"\n💾 df_jogadores salvo em: {CAMINHO_SAIDA}")

