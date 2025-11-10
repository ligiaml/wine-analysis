import pandas as pd
from mainred import mm_normal, mm_gamma, mle_fit

dados = pd.read_csv("winequality-red.csv", sep=';')

x = dados["alcohol"]

mm_norm = mm_normal(x)
mle_norm = mle_fit('normal', x)
mm_gamma_ = mm_gamma(x)
mle_gamma_ = mle_fit('gamma', x)


print("Álcool — Normal (MM):", mm_norm)
print("Álcool — Normal (MLE):", mle_norm)
print("Álcool — Gamma (MM):", mm_gamma_)
print("Álcool — Gamma (MLE):", mle_gamma_)