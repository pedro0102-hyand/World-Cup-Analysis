
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Diretórios
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
GRAFICOS_DIR = os.path.join(BASE_DIR, 'relatorio', 'graficos')
os.makedirs(GRAFICOS_DIR, exist_ok=True)

# Carrega o dataset
df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'df_jogadores.csv'))
df = df.drop_duplicates(subset=['player', 'team'])

# Utiliza apenas colunas numéricas com poucos nulos
df_numerico = df.select_dtypes(include='number')
df_numerico = df_numerico.dropna(axis=1, thresh=int(0.8 * len(df_numerico)))  # colunas com pelo menos 80% preenchidas
df_numerico = df_numerico.fillna(0)

### ===== Funções utilitárias ===== ###
def salvar_barra_metrica(titulo, valores, nome_arquivo):
    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(valores.keys()), y=list(valores.values()), palette='Blues_d')
    plt.title(titulo)
    for i, v in enumerate(valores.values()):
        plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')
    plt.tight_layout()
    caminho = os.path.join(GRAFICOS_DIR, f"estatistico_ml_{nome_arquivo}.png")
    plt.savefig(caminho)
    plt.close()
    print(f"📊 Gráfico salvo: {caminho}")

### ===== 5.1 Previsão de GOLS (Regressão) ===== ###
def regressao_gols(df):
    X = df.drop(columns=['goals'])
    y = df['goals']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelos = {
        "RegLinear": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42)
    }

    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        valores = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "R²": r2_score(y_test, y_pred)
        }

        salvar_barra_metrica(f"{nome} - Previsão de Gols", valores, f"gols_{nome.lower()}")

### ===== 5.2 Previsão de ASSISTÊNCIAS (Regressão) ===== ###
def regressao_assists(df):
    if 'assists' not in df.columns:
        print("⚠️ Coluna 'assists' não disponível.")
        return

    X = df.drop(columns=['assists'])
    y = df['assists']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = RandomForestRegressor(random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    valores = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R²": r2_score(y_test, y_pred)
    }

    salvar_barra_metrica("Random Forest - Previsão de Assistências", valores, "assists_rf")

### ===== 5.3 Classificação: Marcou Gol ou Não ===== ###
def classificacao_gols(df):
    if 'goals' not in df.columns:
        print("⚠️ Coluna 'goals' não disponível.")
        return

    df['gol_binario'] = df['goals'].apply(lambda x: 1 if x > 0 else 0)
    X = df.drop(columns=['goals', 'gol_binario'])
    y = df['gol_binario']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    valores = {
        "Acurácia": acc,
        "Precisão": prec,
        "Recall": rec
    }

    salvar_barra_metrica("Classificação: Marcou Gol", valores, "classificacao_gol")

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão - Gol ou Não")
    caminho = os.path.join(GRAFICOS_DIR, "estatistico_ml_classificacao_confusao.png")
    plt.tight_layout()
    plt.savefig(caminho)
    plt.close()
    print(f"📊 Matriz de confusão salva: {caminho}")

### ===== Execução principal ===== ###
if __name__ == "__main__":
    print("🤖 Iniciando Etapa 5 — Modelagem de Machine Learning...")

    if 'goals' in df_numerico.columns:
        regressao_gols(df_numerico)

    if 'assists' in df_numerico.columns:
        regressao_assists(df_numerico)

    if 'goals' in df.columns:
        classificacao_gols(df_numerico)

    print("✅ Etapa 5 concluída com sucesso.")
