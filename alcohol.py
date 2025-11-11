import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as sp
import statsmodels.api as sm
from mainred import mm_normal, mm_lognormal, mle_fit, mm_gamma

dados = pd.read_csv("winequality-red.csv", sep=';')
x = dados['alcohol']

mm_norm_ = mm_normal(x)
mle_norm_ = mle_fit('norm', x)
mm_lognorm_ = mm_lognormal(x)
mle_lognorm_ = mle_fit('lognorm', x)

def plot_media_mediana_histograma_alcohol(dados):
    """
    Plota histograma dos valores de alcohol, mostrando a distribuiçao no conjunto de dados.
    Nota-se uma distribuiçao assimétrica
    """
    sns.histplot(dados["alcohol"], kde=True, color="#D35269")
    plt.axvline(dados["alcohol"].mean(), color="#361F27", linestyle="--", label="Média")
    plt.axvline(dados["alcohol"].median(), color="#43AA8B", linestyle="--", label="Mediana")
    plt.legend()
    plt.title("Distribuiçaode Alcool com Média e Mediana")
    plt.show()

def ajustar_normal(x):
    """
    Ajusta uma distribuição Normal aos dados de álcool (MLE).
    """
    mean, std = sp.norm.fit(x)
    print(f"Ajuste Normal (MLE): média={mean:.4f} | desvio padrão={std:.4f}")
    return mean, std

def plot_qq_normal(x, mean, std):
    """
    Gera um QQ-plot para verificar o ajuste da distribuição Normal aos dados.
    """
    sm.qqplot(x, dist=sp.norm, loc=mean, scale=std, line='45')
    plt.title("QQ-plot – Distribuição Normal (MLE)")
    plt.show()

def plot_residuos_padronizados(x, mean, std):
    """
    Plota o histograma dos resíduos padronizados de um ajuste estatístico.
    """
    residuos = (x - mean) / std
    plt.figure(figsize=(8, 4))
    plt.hist(residuos, bins=30, color="lightblue", edgecolor="black")
    plt.title("Resíduos padronizados – Álcool (Normal MLE)")
    plt.xlabel("Resíduo padronizado")
    plt.ylabel("Frequência")
    plt.axvline(0, color='black', linestyle='--')
    plt.show()

def detectar_outliers(x, mean, std):
    """
    Identifica e exibe os outliers com base nos resíduos padronizados (> 3 desvios padrão).
    """
    residuos = (x - mean) / std
    outliers = np.where(np.abs(residuos) > 3)[0]
    print(f"Outliers encontrados: {len(outliers)}")
    try:
        print(x.iloc[outliers].values)
    except AttributeError:
        print(np.array(x)[outliers])
    return outliers

def reajuste_sem_outliers(x, outliers):
    """
    Remove os outliers dos dados e realiza um novo ajuste Normal (MLE).
    """
    x_sem_outliers = x.drop(index=x.index[outliers]) if hasattr(x, "drop") else np.delete(x, outliers)
    mean2, std2 = sp.norm.fit(x_sem_outliers)
    print(f"Reajuste sem outliers: média={mean2:.4f} | desvio novo={std2:.4f}")
    return mean2, std2, x_sem_outliers

def plot_comparacao_densidades(x, x_sem_outliers):
    """
    Plota a comparação das densidades dos dados originais e dos dados sem outliers.
    """
    plt.figure(figsize=(8, 5))
    sns.kdeplot(x, label='Original', fill=True, alpha=0.3, color='red')
    sns.kdeplot(x_sem_outliers, label='Sem outliers', fill=True, alpha=0.3, color='green')
    plt.title("Comparação das densidades antes e depois da remoção de outliers (Álcool)")
    plt.xlabel("Álcool (%)")
    plt.legend()
    plt.show()

plot_media_mediana_histograma_alcohol(dados)
mean, std = ajustar_normal(x)
plot_qq_normal(x, mean, std)
plot_residuos_padronizados(x, mean, std)
outliers = detectar_outliers(x, mean, std)
mean2, std2, x_sem_outliers = reajuste_sem_outliers(x, outliers)
plot_comparacao_densidades(x, x_sem_outliers)

print("Álcool — Normal (MM):", mm_norm_)
print("Álcool — Normal (MLE):", mle_norm_)
print("Álcool — LogNormal (MM):", mm_lognorm_)
print("Álcool — LogNormal (MLE):", mle_lognorm_)

def log_likelihood_normal(x, mean, std):
    return np.sum(sp.norm.logpdf(x, mean, std))

def aic_bic(x, mean, std, k=2):
    ll = log_likelihood_normal(x, mean, std)
    n = len(x)
    aic = 2*k - 2*ll
    bic = np.log(n)*k - 2*ll
    return aic, bic

aic1, bic1 = aic_bic(x, mean, std)
aic2, bic2 = aic_bic(x_sem_outliers, mean2, std2)

print("\nAIC e BIC (antes e depois da remoção de outliers):")
print(f"AIC original: {aic1:.2f} | AIC sem outliers: {aic2:.2f}")
print(f"BIC original: {bic1:.2f} | BIC sem outliers: {bic2:.2f}")