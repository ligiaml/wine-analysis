import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    A curva na cor #D35269 mostra a densidade de probabilidade estimada
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
    A curva na cor #E06C9F mostra a densidade de probabilidade estimada
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
    A curva na cor #4A5043 mostra a densidade de probabilidade estimada
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

    """
    sns.histplot(dados["citric acid"], kde=True, color="#4A5043")
    plt.axvline(dados["citric acid"].mean(), color="#F95738", linestyle="--", label="Média")
    plt.axvline(dados["citric acid"].median(), color="#B9D8C2", linestyle="--", label="Mediana")
    plt.legend()
    plt.title("Distribuiçaode acido citrico com Média e Mediana")
    plt.show()

def plot_media_mediana_histograma_sulphates(dados):
    """

    """
    sns.histplot(dados["sulphates"], kde=True, color="#4A5043")
    plt.axvline(dados["sulphates"].mean(), color="#F95738", linestyle="--", label="Média")
    plt.axvline(dados["sulphates"].median(), color="#B9D8C2", linestyle="--", label="Mediana")
    plt.legend()
    plt.title("Distribuiçao de sulphates com Média e Mediana")
    plt.show()
print("moda: ", dados["quality"].mode()[0])

plot_medias_quality(dados) ## A moda da qualidade é 5, indicando que essa é a nota mais comum atribuída aos vinhos
plot_media_mediana_histograma_alcohol(dados)
plot_media_mediana_histograma_volatile_acidity(dados)
plot_media_mediana_histograma_density(dados)
plot_media_mediana_histograma_citric(dados)
plot_media_mediana_histograma_sulphates(dados)
##nota-se que tem muitos dados, dificil analisar visualmente, entao aplicamos agrupamento por qualidade e tiramos as medias
##vinhos com mais alcool tendem a ter qualidade mais alta,
##adicionando a variavel volatile acidity, nota-se que quanto maior a qualidade, maior o nivel de alcool e menor o nivel de acidez
###dar uma olhada em gama e chisquared