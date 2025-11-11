import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, gamma, kendalltau
import warnings
warnings.filterwarnings("ignore")

dados = pd.read_csv("winequality-red.csv", sep=";")
X = dados['alcohol'].values
Y = dados['density'].values

print("="*70)
print("ANÁLISE CONJUNTA COM CÓPULA GAUSSIANA")
print("Variáveis: Álcool (%) x Densidade (g/cm³)")
print("="*70)

mu_x, sigma_x = norm.fit(X)
alpha_y, loc_y, theta_y = gamma.fit(Y, floc=0)

print(f"\nÁlcool ~ Normal(μ={mu_x:.4f}, σ={sigma_x:.4f})")
print(f"Densidade ~ Gamma(α={alpha_y:.4f}, θ={theta_y:.6f})")

def plot_media_mediana_histograma(var, nome, cor):
    sns.histplot(var, kde=True, color=cor)
    plt.axvline(np.mean(var), color="#F95738", linestyle="--", label="Média")
    plt.axvline(np.median(var), color="#B9D8C2", linestyle="--", label="Mediana")
    plt.legend()
    plt.title(f"Distribuição de {nome} com Média e Mediana")
    plt.xlabel(nome)
    plt.show()

plot_media_mediana_histograma(X, "Álcool (%)", "#e74c3c")
plot_media_mediana_histograma(Y, "Densidade (g/cm³)", "#3498db")

U = norm.cdf(X, mu_x, sigma_x)
V = gamma.cdf(Y, alpha_y, loc_y, theta_y)
Z_U = norm.ppf(U)
Z_V = norm.ppf(V)

rho = np.corrcoef(Z_U, Z_V)[0, 1]
tau, _ = kendalltau(X, Y)

print(f"\nCorrelação (ρ): {rho:.4f}")
print(f"Tau de Kendall: {tau:.4f}")
print(f"Interpretação: {'Correlação positiva' if rho > 0 else 'Correlação negativa'}")

n_sim = len(X)
mean = [0, 0]
cov = [[1, rho], [rho, 1]]
Z_sim = np.random.multivariate_normal(mean, cov, n_sim)

U_sim = norm.cdf(Z_sim[:, 0])
V_sim = norm.cdf(Z_sim[:, 1])

X_sim = norm.ppf(U_sim, mu_x, sigma_x)
Y_sim = gamma.ppf(V_sim, alpha_y, loc_y, theta_y)

print(f"\nSimulação concluída: {n_sim} observações")
print(f"Média simulada (Álcool): {X_sim.mean():.4f}")
print(f"Média simulada (Densidade): {Y_sim.mean():.6f}")

def plot_residuos_padronizados(x, mean, std, nome):
    residuos = (x - mean) / std
    plt.figure(figsize=(8,4))
    sns.histplot(residuos, bins=30, kde=True, color="#6baed6")
    plt.axvline(0, color="black", linestyle="--")
    plt.title(f"Resíduos padronizados – {nome}")
    plt.xlabel("Resíduo padronizado")
    plt.ylabel("Frequência")
    plt.show()

plot_residuos_padronizados(X, mu_x, sigma_x, "Álcool (Normal)")
plot_residuos_padronizados(Y, alpha_y * theta_y, np.sqrt(alpha_y) * theta_y, "Densidade (Gamma)")

plt.figure(figsize=(16, 12))

plt.subplot(3, 3, 1)
plt.scatter(X, Y, s=20, alpha=0.5, color='#e74c3c')
plt.title("Dados Observados")
plt.xlabel("Álcool (%)")
plt.ylabel("Densidade (g/cm³)")

plt.subplot(3, 3, 2)
plt.scatter(X_sim, Y_sim, s=20, alpha=0.5, color='#3498db')
plt.title("Dados Simulados (Cópula Gaussiana)")
plt.xlabel("Álcool (%)")
plt.ylabel("Densidade (g/cm³)")

plt.subplot(3, 3, 3)
plt.hist2d(X, Y, bins=30, cmap='Reds')
plt.colorbar(label="Frequência")
plt.title("Densidade Conjunta Observada")

plt.subplot(3, 3, 4)
plt.hist2d(X_sim, Y_sim, bins=30, cmap='Blues')
plt.colorbar(label="Frequência")
plt.title("Densidade Conjunta Simulada")

plt.subplot(3, 3, 5)
sns.kdeplot(X, label="Observado", fill=True, color="#e74c3c", alpha=0.5)
sns.kdeplot(X_sim, label="Simulado", fill=True, color="#3498db", alpha=0.5)
plt.title("Marginal: Álcool")
plt.xlabel("Álcool (%)")
plt.legend()

plt.subplot(3, 3, 6)
sns.kdeplot(Y, label="Observado", fill=True, color="#e74c3c", alpha=0.5)
sns.kdeplot(Y_sim, label="Simulado", fill=True, color="#3498db", alpha=0.5)
plt.title("Marginal: Densidade")
plt.xlabel("Densidade (g/cm³)")
plt.legend()

plt.subplot(3, 3, 7)
plt.scatter(U, V, s=15, alpha=0.6, color="#e74c3c")
plt.title("Cópula Empírica")
plt.xlabel("U (Álcool)")
plt.ylabel("V (Densidade)")
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.subplot(3, 3, 8)
plt.scatter(U_sim, V_sim, s=15, alpha=0.6, color="#3498db")
plt.title("Cópula Gaussiana Ajustada")
plt.xlabel("U (Álcool)")
plt.ylabel("V (Densidade)")
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.subplot(3, 3, 9)
stats.probplot(X, dist="norm", plot=plt)
plt.title("QQ-plot – Álcool ~ Normal")

plt.tight_layout()
plt.savefig("copula_alcool_densidade_completa.png", dpi=300, bbox_inches="tight")
plt.show()

corr_obs = np.corrcoef(X, Y)[0, 1]
corr_sim = np.corrcoef(X_sim, Y_sim)[0, 1]

print("\nCorrelação observada:", round(corr_obs, 4))
print("Correlação simulada :", round(corr_sim, 4))
print("Diferença absoluta  :", abs(corr_obs - corr_sim))

ks_x = stats.kstest(X, lambda x: norm.cdf(x, mu_x, sigma_x))
ks_y = stats.kstest(Y, lambda y: gamma.cdf(y, alpha_y, loc_y, theta_y))
print(f"\nKS - Álcool: estat={ks_x.statistic:.4f}, p={ks_x.pvalue:.4f}")
print(f"KS - Densidade: estat={ks_y.statistic:.4f}, p={ks_y.pvalue:.4f}")

print("\n Análise concluída! Gráfico salvo como 'copula_alcool_densidade_completa.png'")
