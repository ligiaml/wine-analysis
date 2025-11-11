import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as sp
import statsmodels.api as sm
from mainred import mm_normal, mm_lognormal, mle_fit, mm_gamma

dados = pd.read_csv("winequality-red.csv", sep=';')
x = dados['sulphates']

def plot_media_mediana_histograma_sulphates(dados):
    """
    Plota histograma dos valores de sulfato, mostrando a distribuição do conjunto de dados
    """
    sns.histplot(dados["sulphates"], kde=True, color="#4A5043")
    plt.axvline(dados["sulphates"].mean(), color="#F95738", linestyle="--", label="Média")
    plt.axvline(dados["sulphates"].median(), color="#B9D8C2", linestyle="--", label="Mediana")
    plt.legend()
    plt.title("Distribuiçao de sulphates com Média e Mediana")
    plt.show()

mm_gamma_ = mm_gamma(x)
mle_gamma_ = mle_fit('gamma', x)
mm_lognorm_ = mm_lognormal(x)
mle_lognorm_ = mle_fit('lognorm', x)

def ajustar_gamma(x):
    a, loc, scale = sp.gamma.fit(x, floc=0)
    mean = a * scale
    std = np.sqrt(a) * scale
    print(f"Ajuste Gamma (MLE): α={a:.3f}, loc={loc:.3f}, θ={scale:.4f}")
    print(f"Média estimada: {mean:.4f} | Desvio padrão: {std:.4f}")
    return a, loc, scale, mean, std

def plot_qq_gamma(x, a, loc, scale):
    """
    Gera um QQ-plot para verificar o ajuste de uma distribuição Gamma aos dados
    """
    sm.qqplot(x, dist=sp.gamma, distargs=(a,), loc=loc, scale=scale, line='45')
    plt.title("QQ-plot – Distribuição Gama (MLE)")
    plt.show()

def plot_residuos_padronizados(x, mean, std):
    """
    Plota o histograma dos resíduos padronizados de um ajuste estatístico
    """
    residuos = (x - mean) / std

    plt.figure(figsize=(8, 4))
    plt.hist(residuos, bins=30, color="lightblue", edgecolor="black")
    plt.title("Resíduos padronizados – Sulfatos (Gamma MLE)")
    plt.xlabel("Resíduo padronizado")
    plt.ylabel("Frequência")
    plt.axvline(0, color='black', linestyle='--')
    plt.show()

def detectar_outliers(x, mean, std):
    """
    Identifica e exibe os outliers com base nos resíduos padronizados (> 3 desvios padrão)
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
    Remove os outliers dos dados e realiza um novo ajuste Gamma (MLE)
    """
    x_sem_outliers = x.drop(index=x.index[outliers]) if hasattr(x, "drop") else np.delete(x, outliers)
    
    a2, loc2, scale2 = sp.gamma.fit(x_sem_outliers, floc=0)
    mean2 = a2 * scale2
    std2 = np.sqrt(a2) * scale2

    print(f"Reajuste sem outliers: α={a2:.3f}, θ={scale2:.4f}")
    print(f"Média nova: {mean2:.4f} | Desvio novo: {std2:.4f}")

    return a2, loc2, scale2, mean2, std2, x_sem_outliers

def plot_comparacao_densidades(x, x_sem_outliers):
    """
    Plota a comparação das densidades dos dados originais e dos dados sem outliers.
    """
    plt.figure(figsize=(8, 5))
    sns.kdeplot(x, label='Original', fill=True, alpha=0.3, color='red')
    sns.kdeplot(x_sem_outliers, label='Sem outliers', fill=True, alpha=0.3, color='green')
    plt.title("Comparação das densidades antes e depois da remoção de outliers")
    plt.xlabel("Sulfatos (g/dm³)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_media_mediana_histograma_sulphates(dados)
    a, loc, scale, mean, std = ajustar_gamma(x)
    plot_qq_gamma(x, a, loc, scale)
    plot_residuos_padronizados(x, mean, std)
    outliers = detectar_outliers(x, mean, std)
    a2, loc2, scale2, mean2, std2, x_sem_outliers = reajuste_sem_outliers(x, outliers)
    plot_comparacao_densidades(x, x_sem_outliers)

def log_likelihood_gamma(x, a, loc, scale):
    return np.sum(sp.gamma.logpdf(x, a, loc, scale))

def aic_bic(x, a, loc, scale, k=3):
    ll = log_likelihood_gamma(x, a, loc, scale)
    n = len(x)
    aic = 2*k - 2*ll
    bic = np.log(n)*k - 2*ll
    return aic, bic

print("Sulfatos — Gamma (MM):", mm_gamma_)
print("Sulfatos — Gamma (MLE):", mle_gamma_)
print("Sulfatos — LogNormal (MM):", mm_lognorm_)
print("Sulfatos — LogNormal (MLE):", mle_lognorm_)

aic1, bic1 = aic_bic(x, a, loc, scale)
aic2, bic2 = aic_bic(x_sem_outliers, a2, loc2, scale2)

print("\nAIC e BIC (antes e depois da remoção de outliers):")
print(f"AIC original: {aic1:.2f} | AIC sem outliers: {aic2:.2f}")
print(f"BIC original: {bic1:.2f} | BIC sem outliers: {bic2:.2f}")