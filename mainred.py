import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sp
import seaborn as sns
import numpy as np


dados = pd.read_csv('winequality-red.csv', sep=";")

def plot_medias_quality(dados):
    """
    plotar graficos mostrando a media das variaveis em relaçao a qualidade do vinho, para ver tendencias
    """

    media = dados.groupby("quality")[["alcohol", "volatile acidity", "citric acid", "sulphates", "density"]].mean()
    fig, axes = plt.subplots(1,5, figsize=(20,4))

    colunas = ["alcohol", "volatile acidity", "citric acid", "sulphates","density"]
    cores = ["pink", "green", "black", "orange", "blue"]
    titulos = ["Álcool x Qualidade","Acidez Volátil x Qualidade","Ácido Cítrico x Qualidade","Sulfatos x Qualidade","Densidade x Qualidade"]

    for x, coluna in enumerate(colunas):
        media[coluna].plot(kind="bar", ax=axes[x], color=cores[x], alpha = 0.4)
        axes[x].set_title(titulos[x])
        axes[x].set_ylabel("Média")
        
        if coluna == "density":
            axes[x].set_ylim(0.99, 1.005)
    
    plt.tight_layout()
    plt.show()
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
def plot_media_mediana_histograma_volatile_acidity(dados):
    """
    Plota histograma dos valores de acidez volatil, mostrando a distribuiçao no conjunto de dados.
    Nota-se uma distribuiçao assimétrica
    """
    sns.histplot(dados["volatile acidity"], kde=True, color="#E06C9F")
    plt.axvline(dados["volatile acidity"].mean(), color="#B5BFA1", linestyle="--", label="Média")
    plt.axvline(dados["volatile acidity"].median(), color="#6E9887", linestyle="--", label="Mediana")
    plt.legend()
    plt.title("Distribuiçaode Acidez Volatil com Média e Mediana")
    plt.show()
def plot_media_mediana_histograma_density(dados):
    """
    Plota histograma dos valores de densidade, mostrando a distribuiçao no conjunto de dados.
    Nota-se uma distribuiçao simétrica, distribuição normal
    """
    sns.histplot(dados["density"], kde=True, color="#4A5043")
    plt.axvline(dados["density"].mean(), color="#F95738", linestyle="--", label="Média")
    plt.axvline(dados["density"].median(), color="#B9D8C2", linestyle="--", label="Mediana")
    plt.legend()
    plt.title("Distribuiçaode Densidade com Média e Mediana")
    plt.show()
def plot_media_mediana_histograma_citric(dados):
    """
    Plota histograma dos valores de acido citrico, mostrando a distribuição do conjunto de dados
    """
    sns.histplot(dados["citric acid"], kde=True, color="#4A5043")
    plt.axvline(dados["citric acid"].mean(), color="#F95738", linestyle="--", label="Média")
    plt.axvline(dados["citric acid"].median(), color="#B9D8C2", linestyle="--", label="Mediana")
    plt.legend()
    plt.title("Distribuiçaode Acido Citrico com Média e Mediana")
    plt.show()
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
def plot_media_histograma_fixed_acidity(dados):
    sns.histplot(dados["fixed acidity"], kde=True, color="#4A5043")
    plt.axvline(dados["fixed acidity"].mean(), color="#F95738", linestyle="--", label="Média")
    plt.axvline(dados["fixed acidity"].median(), color="#B9D8C2", linestyle="--", label="Mediana")
    plt.legend()
    plt.title("Distribuiçao de acidez fixa com Média e Mediana")
    plt.show()
print("moda: ", dados["quality"].mode()[0])

plot_media_mediana_histograma_alcohol(dados)
plot_media_mediana_histograma_density(dados)
plot_media_mediana_histograma_sulphates(dados)
plot_media_histograma_fixed_acidity(dados)
##nota-se que tem muitos dados, dificil analisar visualmente, entao aplicamos agrupamento por qualidade e tiramos as medias
##vinhos com mais alcool tendem a ter qualidade mais alta,
##adicionando a variavel volatile acidity, nota-se que quanto maior a qualidade, maior o nivel de alcool e menor o nivel de acidez
###dar uma olhada em gama e chisquared

plt.boxplot(dados["alcohol"])
plt.title("boxplot alcohol")
plt.show()
plt.boxplot(dados["density"])
plt.title("boxplot density")
plt.show()
plt.boxplot(dados["sulphates"])
plt.title("boxplor sulfatos")
plt.show()
plt.boxplot(dados["fixed acidity"])
plt.title("boxplot acidez fixa")
plt.show()

############################

def mm_normal(x):
    """calcula média e desvio padrão (método dos momentos da Normal)"""
    return {'mu': np.mean(x), 'sigma': np.std(x, ddof=0)}
def mm_gamma(x):
    m=np.mean(x)
    v=np.var(x,ddof=0)
    alpha=m**2 /v
    beta = v/m
    return {'a': alpha, 'scale':beta}
def mm_lognormal(x):
    m=np.mean(x)
    v=np.var(x, ddof=0)
    phi=np.log(1+v/m**2)
    sigma=np.sqrt(phi)
    mu=np.log(m) - 0.5 * phi
    return{'mu':mu, 'sigma':sigma}
def mle_fit(dist_name, x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("Nenhum dado válido encontrado (todos eram NaN ou infinitos).")
    if dist_name in ['normal', 'norm']:
        mu, sigma = sp.norm.fit(x)
        return {'mu': mu, 'sigma': sigma}
    elif dist_name == 'gamma':
        a, loc, scale = sp.gamma.fit(x, floc=0)
        return {'a': a, 'loc': loc, 'scale': scale}
    elif dist_name == 'lognorm':
        s, loc, scale = sp.lognorm.fit(x, floc=0)
        return {'s': s, 'loc': loc, 'scale': scale}
    else:
        raise ValueError(f"Distribuição não suportada: {dist_name}")

##########################

def compara_ajustes(x, nome_var, dists):
    """
    Gera histograma e curvas ajustadas para as distribuições em 'dists'.
    dists = lista de strings ('normal', 'gamma', 'lognorm')
    """
    x=x.dropna()
    plt.figure(figsize=(8,5))
    plt.hist(x, bins=30, density=True, alpha=0.6, color='gray', label='Dados observados')
    xs=np.linspace(x.min(), x.max(), 300)

    for dist_name in dists:
        params = mle_fit(dist_name, x)
        if dist_name == 'normal':
            pdf = sp.norm.pdf(xs, params['mu'], params['sigma'])
            label = f"Normal (μ={params['mu']:.3f}, σ={params['sigma']:.3f})"
        elif dist_name == 'gamma':
            pdf = sp.gamma.pdf(xs, params['a'], params['loc'], params['scale'])
            label = f"Gamma (α={params['a']:.2f}, θ={params['scale']:.3f})"
        elif dist_name == 'lognorm':
            pdf = sp.lognorm.pdf(xs, params['s'], params['loc'], params['scale'])
            label = f"LogNorm (μ={np.log(params['scale']):.3f}, σ={params['s']:.3f})"
        plt.plot(xs, pdf, label=label)
        plt.title(f"Comparação: dados observados x ajustes ({nome_var})")
        plt.xlabel(nome_var)
        plt.ylabel("Densidade")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
compara_ajustes(dados["alcohol"], "Álcool", ["normal", "gamma"])
compara_ajustes(dados["density"], "Densidade", ["normal", "lognorm"])
compara_ajustes(dados["sulphates"], "Sulfatos", ["gamma", "lognorm"])
