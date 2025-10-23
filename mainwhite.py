import pandas as pd
import matplotlib.pyplot as plt


dados = pd.read_csv('winequality-white.csv', sep=";")

media_values = dados.groupby("quality")[["alcohol", "volatile acidity", "citric acid", "sulphates", "residual sugar"]].mean()
fig, axes = plt.subplots(1,5, figsize=(20,4))

media_values["alcohol"].plot(kind="bar", ax = axes[0], color = "pink", alpha=0.6)
axes[0].set_title("Alcool x Qualidade")
axes[0].set_ylabel("Media")

media_values["volatile acidity"].plot(kind="bar", ax=axes[1], color="green", alpha=0.6)
axes[1].set_title("Acidez volatil x Qualidade")
axes[1].set_ylabel("Media")

media_values["citric acid"].plot(kind="bar", ax=axes[2], color="black", alpha=0.6)
axes[2].set_title("Acidez Citrica x Qualidade")
axes[2].set_ylabel("Media")

media_values["sulphates"].plot(kind="bar", ax=axes[3], color="orange", alpha=0.6 )
axes[3].set_title("Sulfatos x Qualidade")
axes[3].set_ylabel("Media")

media_values["residual sugar"].plot(kind="bar", ax=axes[4], color="blue", alpha=0.6 )
axes[4].set_title("Açucar residual x Qualidade")
axes[4].set_ylabel("Media")

plt.tight_layout()
plt.show()
##plotando nivel de alcool e nivel de qualidade - tem correlaçao (talvez?)


##nota-se que tem muitos dados, dificil analisar visualmente, entao aplicamos agrupamento por qualidade e tiramos as medias
##vinhos com mais alcool tendem a ter qualidade mais alta,
##adicionando a variavel volatile acidity, nota-se que quanto maior a qualidade, maior o nivel de alcool e menor o nivel de acidez
###dar uma olhada em gama e chisquared