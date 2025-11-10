import pandas as pd
from mainred import mm_normal, mm_lognormal, mle_fit, mm_gamma

dados = pd.read_csv("winequality-red.csv", sep=';')
x = dados['sulphates']

mm_gamma_ = mm_gamma(x)
mle_gamma_ = mle_fit('gamma', x)
mm_lognorm_ = mm_lognormal(x)
mle_lognorm_ = mle_fit('lognorm', x)

print("Sulfatos — Gamma (MM):", mm_gamma_)
print("Sulfatos — Gamma (MLE):", mle_gamma_)
print("Sulfatos — LogNormal (MM):", mm_lognorm_)
print("Sulfatos — LogNormal (MLE):", mle_lognorm_)
