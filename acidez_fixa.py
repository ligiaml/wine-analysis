import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sp
import statsmodels.api as sm
from mainred import mm_gamma, mm_lognormal, mle_fit

dados = pd.read_csv("winequality-red.csv", sep=';')
x = dados['fixed acidity'].dropna()


def plot_media_histograma_fixed_acidity(dados):
    """Plota histograma da acidez fixa com média e mediana"""
    sns.histplot(dados["fixed acidity"], kde=True, color="#4A5043")
    plt.axvline(dados["fixed acidity"].mean(), color="#F95738", linestyle="--", label="Média")
    plt.axvline(dados["fixed acidity"].median(), color="#B9D8C2", linestyle="--", label="Mediana")
    plt.legend()
    plt.title("Distribuição da Acidez Fixa com Média e Mediana")
    plt.xlabel("Acidez Fixa (g/dm³)")
    plt.show()

def ajustar_gamma(x):
    """Ajusta a distribuição Gamma via MLE e calcula média e desvio"""
    a, loc, scale = sp.gamma.fit(x, floc=0)
    mean = a * scale
    std = np.sqrt(a) * scale
    print(f"Ajuste Gamma (MLE): α={a:.3f}, loc={loc:.3f}, θ={scale:.4f}")
    print(f"Média estimada: {mean:.4f} | Desvio padrão: {std:.4f}")
    return a, loc, scale, mean, std

def plot_qq_gamma(x, a, loc, scale):
    """Gera QQ-plot para avaliar aderência da distribuição Gamma"""
    sm.qqplot(x, dist=sp.gamma, distargs=(a,), loc=loc, scale=scale, line='45')
    plt.title("QQ-plot – Acidez Fixa (Gamma MLE)")
    plt.show()

def plot_residuos_padronizados(x, mean, std):
    """Plota histograma dos resíduos padronizados"""
    residuos = (x - mean) / std
    plt.hist(residuos, bins=30, color="lightblue", edgecolor="black")
    plt.title("Resíduos Padronizados – Acidez Fixa (Gamma MLE)")
    plt.xlabel("Resíduo Padronizado")
    plt.ylabel("Frequência")
    plt.axvline(0, color='black', linestyle='--')
    plt.show()
    return residuos

def detectar_outliers(x, mean, std):
    """Identifica outliers com base em resíduos > 3 desvios padrão"""
    residuos = (x - mean) / std
    outliers = np.where(np.abs(residuos) > 3)[0]
    print(f"Outliers encontrados: {len(outliers)}")
    try:
        print(x.iloc[outliers].values)
    except AttributeError:
        print(np.array(x)[outliers])
    return outliers

def reajuste_sem_outliers(x, outliers):
    """Remove outliers e refaz ajuste Gamma"""
    x_sem_outliers = x.drop(index=x.index[outliers]) if hasattr(x, "drop") else np.delete(x, outliers)
    a2, loc2, scale2 = sp.gamma.fit(x_sem_outliers, floc=0)
    mean2 = a2 * scale2
    std2 = np.sqrt(a2) * scale2
    print(f"Reajuste sem outliers: α={a2:.3f}, θ={scale2:.4f}")
    print(f"Média nova={mean2:.4f} | Desvio novo={std2:.4f}")
    return a2, loc2, scale2, mean2, std2, x_sem_outliers

def plot_comparacao_densidades(x, x_sem_outliers):
    """Compara densidades antes e depois da remoção dos outliers"""
    plt.figure(figsize=(8,5))
    sns.kdeplot(x, label='Original', fill=True, color='red', alpha=0.3)
    sns.kdeplot(x_sem_outliers, label='Sem outliers', fill=True, color='green', alpha=0.3)
    plt.title("Comparação das Densidades – Acidez Fixa (Gamma)")
    plt.xlabel("Acidez Fixa (g/dm³)")
    plt.legend()
    plt.show()

plot_media_histograma_fixed_acidity(dados)

a, loc, scale, mean, std = ajustar_gamma(x)
plot_qq_gamma(x, a, loc, scale)
plot_residuos_padronizados(x, mean, std)
outliers = detectar_outliers(x, mean, std)
a2, loc2, scale2, mean2, std2, x_sem_outliers = reajuste_sem_outliers(x, outliers)
plot_comparacao_densidades(x, x_sem_outliers)

mm_gamma_ = mm_gamma(x)
mle_gamma_ = mle_fit('gamma', x)
mm_lognorm_ = mm_lognormal(x)
mle_lognorm_ = mle_fit('lognorm', x)

print("\nAcidez Fixa — Gamma (MM):", mm_gamma_)
print("Acidez Fixa — Gamma (MLE):", mle_gamma_)
print("Acidez Fixa — LogNormal (MM):", mm_lognorm_)
print("Acidez Fixa — LogNormal (MLE):", mle_lognorm_)

def log_likelihood_gamma(x, a, loc, scale):
    return np.sum(sp.gamma.logpdf(x, a, loc, scale))

def aic_bic_gamma(x, a, loc, scale, k=3):
    ll = log_likelihood_gamma(x, a, loc, scale)
    n = len(x)
    aic = 2*k - 2*ll
    bic = np.log(n)*k - 2*ll
    return aic, bic

aic1, bic1 = aic_bic_gamma(x, a, loc, scale)
aic2, bic2 = aic_bic_gamma(x_sem_outliers, a2, loc2, scale2)

print("\nAIC e BIC (antes e depois da remoção de outliers):")
print(f"AIC original: {aic1:.2f} | AIC sem outliers: {aic2:.2f}")
print(f"BIC original: {bic1:.2f} | BIC sem outliers: {bic2:.2f}")
