
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind

# Caminhos
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
GRAFICOS_DIR = os.path.join(BASE_DIR, 'relatorio', 'graficos')
os.makedirs(GRAFICOS_DIR, exist_ok=True)

# Carrega o DataFrame unificado
df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'df_jogadores.csv'))

# Limpeza básica
df = df.drop_duplicates(subset=['player', 'team'])

### ===== 4.1 Correlação ===== ###
def gerar_heatmap_correlacao(df):
    colunas_numericas = df.select_dtypes(include='number').dropna(axis=1, how='any')
    correlacoes = colunas_numericas.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlacoes, cmap='coolwarm', annot=True, fmt=".2f", square=True)
    plt.title('Mapa de Correlação entre Métricas')
    caminho = os.path.join(GRAFICOS_DIR, 'estatistico_correlacao.png')
    plt.tight_layout()
    plt.savefig(caminho)
    plt.close()
    print(f"📊 Heatmap de correlação salvo em: {caminho}")

### ===== 4.2 Testes de hipótese ===== ###
def testes_hipotese(df):
    # Exemplo: atacantes vs zagueiros em xG
    atacantes = df[df['position'] == 'FW']['xg'].dropna()
    zagueiros = df[df['position'] == 'DF']['xg'].dropna()
    
    if not atacantes.empty and not zagueiros.empty:
        t_stat, p_val = ttest_ind(atacantes, zagueiros, equal_var=False)
        print(f"🔬 Teste de hipótese (FW vs DF em xG): t={t_stat:.2f}, p={p_val:.4f}")
    else:
        print("⚠️ Não há dados suficientes para o teste de hipótese FW vs DF.")

    # Exemplo 2: jogadores com >300 min vs <300 min em gols
    grupo_mais = df[df['minutes_90s'] > 3]['goals'].dropna()
    grupo_menos = df[df['minutes_90s'] <= 3]['goals'].dropna()
    
    if not grupo_mais.empty and not grupo_menos.empty:
        t_stat, p_val = ttest_ind(grupo_mais, grupo_menos, equal_var=False)
        print(f"🔬 Teste de hipótese (Min >3 vs <=3 em gols): t={t_stat:.2f}, p={p_val:.4f}")
    else:
        print("⚠️ Não há dados suficientes para o teste de minutos.")

### ===== 4.3 PCA ===== ###
def realizar_pca(df):
    colunas_numericas = df.select_dtypes(include='number')
    colunas_filtradas = colunas_numericas.dropna(axis=1, thresh=int(0.8 * len(colunas_numericas)))  # pelo menos 80% não nulo

    if colunas_filtradas.shape[1] < 2:
        print("⚠️ PCA cancelado: menos de 2 colunas numéricas válidas após limpeza.")
        return

    pca = PCA(n_components=2)
    componentes = pca.fit_transform(colunas_filtradas.fillna(0))
    df_pca = pd.DataFrame(componentes, columns=['PC1', 'PC2'])
    df_pca['player'] = df['player'].values[:len(df_pca)]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', s=50, alpha=0.8)
    plt.title('PCA: Redução de Dimensionalidade dos Jogadores')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    caminho = os.path.join(GRAFICOS_DIR, 'estatistico_pca.png')
    plt.tight_layout()
    plt.savefig(caminho)
    plt.close()
    print(f"📊 Gráfico PCA salvo em: {caminho}")



### ===== 4.4 Clustering ===== ###
def clustering_kmeans(df, n_clusters=4):
    colunas_numericas = df.select_dtypes(include='number')
    colunas_filtradas = colunas_numericas.dropna(axis=1, thresh=int(0.8 * len(colunas_numericas)))

    if colunas_filtradas.shape[1] < 2:
        print("⚠️ K-Means cancelado: menos de 2 colunas numéricas válidas após limpeza.")
        return

    X = colunas_filtradas.fillna(0)
    modelo = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    clusters = modelo.fit_predict(X)

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X)

    df_clusters = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
    df_clusters['cluster'] = clusters

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_clusters, x='PC1', y='PC2', hue='cluster', palette='tab10', s=60)
    plt.title('Clustering K-Means com PCA')
    caminho = os.path.join(GRAFICOS_DIR, 'estatistico_clusters.png')
    plt.tight_layout()
    plt.savefig(caminho)
    plt.close()
    print(f"📊 Clusters salvos em: {caminho}")


### ===== Execução ===== ###
if __name__ == "__main__":
    print("🔎 Iniciando Etapa 4 — Modelagem Estatística...")

    gerar_heatmap_correlacao(df)
    testes_hipotese(df)
    realizar_pca(df)
    clustering_kmeans(df)

    print("✅ Etapa 4 concluída com sucesso.")
