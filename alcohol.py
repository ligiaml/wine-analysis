import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import gamma
from mainred import dados

alcohol = dados['alcohol']

def dist_normal_alcohol(alcochol):
    mu = alcohol.mean()
    sigma = alcohol.std()

    x = np.linspace(alcohol.min(), alcohol.max(), 100)
    y = norm.pdf(x,mu,sigma)

    plt.hist(alcohol, bins=20, density=True, alpha=0.6, color='#553E4E', edgecolor='black')
    plt.plot(x, y, 'r', lw=2)
    plt.title('Distribuição Normal - Alcohol')
    plt.xlabel('Alcohol')
    plt.ylabel('Densidade')
    plt.show()

dist_normal_alcohol(alcohol)