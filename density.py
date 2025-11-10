import pandas as pd
from mainred import mm_normal, mm_lognormal, mle_fit

dados = pd.read_csv("winequality-red.csv", sep=';')

x = dados['density']

mm_norm = mm_normal(x)
mle_norm = mle_fit('normal', x)
mm_lognorm_ = mm_lognormal(x)
mle_lognorm_ = mle_fit('lognorm', x)

print("Densidade — Normal (MM):", mm_norm)
print("Densidade — Normal (MLE):", mle_norm)
print("Densidade — LogNormal (MM):", mm_lognorm_)
print("Densidade — LogNormal (MLE):", mle_lognorm_)

