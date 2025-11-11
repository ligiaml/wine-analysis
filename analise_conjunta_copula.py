import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, gamma
import warnings
warnings.filterwarnings('ignore')


dados = pd.read_csv("winequality-red.csv", sep=';')

# Selecionar as duas variáveis
X = dados['alcohol'].values
Y = dados['sulphates'].values

print("="*60)
print("ANÁLISE CONJUNTA COM CÓPULAS")
print("Variáveis: Alcohol x Sulphates")
print("="*60)

# ==============================================================
# 1. AJUSTAR DISTRIBUIÇÕES MARGINAIS
# ==============================================================
print("\n1. AJUSTE DAS DISTRIBUIÇÕES MARGINAIS")
print("-"*60)

# Alcohol ~ Normal
mu_x, sigma_x = norm.fit(X)
print(f"Alcohol ~ Normal(μ={mu_x:.4f}, σ={sigma_x:.4f})")

# Sulphates ~ Gamma
params_y = gamma.fit(Y, floc=0)
alpha_y, loc_y, theta_y = params_y
print(f"Sulphates ~ Gamma(α={alpha_y:.4f}, θ={theta_y:.4f})")

# ==============================================================
# 2. TRANSFORMAÇÃO PARA UNIFORMES (Probability Integral Transform)
# ==============================================================
print("\n2. TRANSFORMAÇÃO PARA UNIFORMES")
print("-"*60)

# Transformar dados para uniformes [0,1] usando as CDFs marginais
U = norm.cdf(X, mu_x, sigma_x)
V = gamma.cdf(Y, alpha_y, loc_y, theta_y)

print(f"U (de Alcohol): min={U.min():.4f}, max={U.max():.4f}")
print(f"V (de Sulphates): min={V.min():.4f}, max={V.max():.4f}")

# ==============================================================
# 3. AJUSTAR CÓPULA GAUSSIANA
# ==============================================================
print("\n3. AJUSTE DA CÓPULA GAUSSIANA")
print("-"*60)

# Transformar uniformes para normais padrão
Z_U = norm.ppf(U)
Z_V = norm.ppf(V)

# Calcular correlação (parâmetro da cópula Gaussiana)
rho = np.corrcoef(Z_U, Z_V)[0, 1]
print(f"Parâmetro da cópula (ρ): {rho:.4f}")
print(f"Interpretação: {'Dependência positiva' if rho > 0 else 'Dependência negativa'}")

# Tau de Kendall (medida não-paramétrica)
from scipy.stats import kendalltau
tau, _ = kendalltau(X, Y)
print(f"Tau de Kendall: {tau:.4f}")

# ==============================================================
# 4. SIMULAR DADOS DA CÓPULA
# ==============================================================
print("\n4. SIMULAÇÃO DE DADOS DA CÓPULA")
print("-"*60)

n_sim = len(X)

# Gerar da cópula Gaussiana bivariada
mean = [0, 0]
cov = [[1, rho], [rho, 1]]
Z_sim = np.random.multivariate_normal(mean, cov, n_sim)

# Transformar para uniformes
U_sim = norm.cdf(Z_sim[:, 0])
V_sim = norm.cdf(Z_sim[:, 1])

# Transformar para as escalas originais
X_sim = norm.ppf(U_sim, mu_x, sigma_x)
Y_sim = gamma.ppf(V_sim, alpha_y, loc_y, theta_y)

print(f"Dados simulados: {n_sim} observações")
print(f"X_sim (alcohol): média={X_sim.mean():.4f}, desvio={X_sim.std():.4f}")
print(f"Y_sim (sulphates): média={Y_sim.mean():.4f}, desvio={Y_sim.std():.4f}")

# ==============================================================
# 5. VISUALIZAÇÕES
# ==============================================================
print("\n5. GERANDO GRÁFICOS COMPARATIVOS...")
print("-"*60)

fig = plt.figure(figsize=(16, 12))

# ===== GRÁFICO 1: Scatter plots comparativos =====
ax1 = plt.subplot(3, 3, 1)
plt.scatter(X, Y, alpha=0.5, s=20, c='#e74c3c')
plt.xlabel('Alcohol (%)', fontsize=10)
plt.ylabel('Sulphates (g/dm³)', fontsize=10)
plt.title('Dados Observados', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

ax2 = plt.subplot(3, 3, 2)
plt.scatter(X_sim, Y_sim, alpha=0.5, s=20, c='#3498db')
plt.xlabel('Alcohol (%)', fontsize=10)
plt.ylabel('Sulphates (g/dm³)', fontsize=10)
plt.title('Dados Simulados (Cópula Gaussiana)', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# ===== GRÁFICO 2: Densidades Conjuntas (Contorno) =====
ax3 = plt.subplot(3, 3, 3)
plt.hist2d(X, Y, bins=30, cmap='Reds', alpha=0.8)
plt.colorbar(label='Frequência')
plt.xlabel('Alcohol (%)', fontsize=10)
plt.ylabel('Sulphates (g/dm³)', fontsize=10)
plt.title('Densidade Conjunta Observada', fontsize=12, fontweight='bold')

ax4 = plt.subplot(3, 3, 4)
plt.hist2d(X_sim, Y_sim, bins=30, cmap='Blues', alpha=0.8)
plt.colorbar(label='Frequência')
plt.xlabel('Alcohol (%)', fontsize=10)
plt.ylabel('Sulphates (g/dm³)', fontsize=10)
plt.title('Densidade Conjunta Simulada', fontsize=12, fontweight='bold')

# ===== GRÁFICO 3: Distribuições Marginais =====
ax5 = plt.subplot(3, 3, 5)
plt.hist(X, bins=30, density=True, alpha=0.6, color='#e74c3c', label='Observado')
plt.hist(X_sim, bins=30, density=True, alpha=0.6, color='#3498db', label='Simulado')
plt.xlabel('Alcohol (%)', fontsize=10)
plt.ylabel('Densidade', fontsize=10)
plt.title('Marginal: Alcohol', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

ax6 = plt.subplot(3, 3, 6)
plt.hist(Y, bins=30, density=True, alpha=0.6, color='#e74c3c', label='Observado')
plt.hist(Y_sim, bins=30, density=True, alpha=0.6, color='#3498db', label='Simulado')
plt.xlabel('Sulphates (g/dm³)', fontsize=10)
plt.ylabel('Densidade', fontsize=10)
plt.title('Marginal: Sulphates', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# ===== GRÁFICO 4: Cópula no espaço [0,1] =====
ax7 = plt.subplot(3, 3, 7)
plt.scatter(U, V, alpha=0.5, s=20, c='#e74c3c')
plt.xlabel('U (de Alcohol)', fontsize=10)
plt.ylabel('V (de Sulphates)', fontsize=10)
plt.title('Cópula Empírica', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)

ax8 = plt.subplot(3, 3, 8)
plt.scatter(U_sim, V_sim, alpha=0.5, s=20, c='#3498db')
plt.xlabel('U (de Alcohol)', fontsize=10)
plt.ylabel('V (de Sulphates)', fontsize=10)
plt.title('Cópula Gaussiana Ajustada', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)

# ===== GRÁFICO 5: QQ-plot das marginais =====
ax9 = plt.subplot(3, 3, 9)
stats.probplot(X, dist="norm", plot=plt)
plt.title('QQ-plot: Alcohol ~ Normal', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analise_copula_completa.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================
# 6. MÉTRICAS DE QUALIDADE DO AJUSTE
# ==============================================================
print("\n6. MÉTRICAS DE QUALIDADE DO AJUSTE")
print("-"*60)

# Correlação de Pearson
corr_obs = np.corrcoef(X, Y)[0, 1]
corr_sim = np.corrcoef(X_sim, Y_sim)[0, 1]
print(f"Correlação de Pearson (observada): {corr_obs:.4f}")
print(f"Correlação de Pearson (simulada): {corr_sim:.4f}")
print(f"Diferença: {abs(corr_obs - corr_sim):.4f}")

# Teste de Kolmogorov-Smirnov para as marginais
ks_x = stats.kstest(X, lambda x: norm.cdf(x, mu_x, sigma_x))
ks_y = stats.kstest(Y, lambda y: gamma.cdf(y, alpha_y, loc_y, theta_y))
print(f"\nTeste KS - Alcohol: estatística={ks_x.statistic:.4f}, p-valor={ks_x.pvalue:.4f}")
print(f"Teste KS - Sulphates: estatística={ks_y.statistic:.4f}, p-valor={ks_y.pvalue:.4f}")

print("\n" + "="*60)
print("ANÁLISE CONCLUÍDA!")
print("Gráfico salvo como: analise_copula_completa.png")
print("="*60)