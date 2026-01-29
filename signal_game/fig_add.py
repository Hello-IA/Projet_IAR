# --- Boxplot de l’écart à la courbe IB (distance euclidienne point→courbe) ---
# Inspiré de ton fig3.py (scatter + IB curve) et de la définition Inef (Eq. 3) de l’article.

from accuracy_complexite import *   # cumpute_humain_langues, cumpute_nn_langues, solve_ib, etc.
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# 1) Charger tes points
# -----------------------
# Langues humaines (WCS)
results_wcs = np.array(cumpute_humain_langues("term.txt"))  # colonnes: [name, I(M;W), I(U;W)]
wcs_points = results_wcs[:, 1:3].astype(float)

# Langues NN par percentile
perc = (20, 40, 60, 80)
all_nn_points = {}
for p in perc:
    r = np.array(cumpute_nn_langues(f"fig3_bis/percentile{p}", 0, 60))  # adapte le path si besoin
    all_nn_points[p] = r[:, 1:3].astype(float)

# -----------------------
# 2) Construire la courbe IB
# -----------------------
betas = np.linspace(1.0, 1.15, 50)

K = 20
q_w_m = np.random.rand(U, K)
q_w_m /= q_w_m.sum(axis=1, keepdims=True)

ib_points = []
for beta in betas:
    I_MW, I_UW, q_w_m = solve_ib(beta, q_w_m, K=K)
    ib_points.append((I_MW, I_UW))

ib_points = np.array(ib_points, dtype=float)
ib_points = ib_points[np.argsort(ib_points[:, 0])]  # tri par complexité
ib_x, ib_y = ib_points[:, 0], ib_points[:, 1]

# ---------------------------------------------------------
# 3) Distance point → polyline (courbe IB) (plus précis)
# ---------------------------------------------------------
def dist_point_to_polyline(P, curve_xy):
    """
    P: (N,2) points
    curve_xy: (M,2) points (polyline)
    Retourne: (N,) distance euclidienne minimale point→segments.
    """
    P = np.asarray(P, dtype=float)
    C = np.asarray(curve_xy, dtype=float)

    A = C[:-1]          # (M-1,2)
    B = C[1:]           # (M-1,2)
    AB = B - A          # (M-1,2)
    AB2 = (AB**2).sum(axis=1)  # (M-1,)

    # Pour vectoriser: on calcule pour chaque point P[i] sa distance à tous les segments
    # proj = A + t*(B-A) avec t clampé dans [0,1]
    dmin = np.empty(P.shape[0], dtype=float)

    for i in range(P.shape[0]):
        AP = P[i] - A                      # (M-1,2)
        t = (AP * AB).sum(axis=1) / AB2    # (M-1,)
        t = np.clip(t, 0.0, 1.0)
        proj = A + (t[:, None] * AB)       # (M-1,2)
        d2 = ((P[i] - proj) ** 2).sum(axis=1)
        dmin[i] = np.sqrt(d2.min())
    return dmin

ib_curve_xy = ib_points

# Distances (écart à la courbe IB)
dist_wcs = dist_point_to_polyline(wcs_points, ib_curve_xy)

dist_nn = {}
for p in perc:
    dist_nn[p] = dist_point_to_polyline(all_nn_points[p], ib_curve_xy)

# -----------------------
# 4) Boxplot
# -----------------------
data = [dist_nn[p] for p in perc] + [dist_wcs]
labels = [str(p) for p in perc] + ["WCS (human)"]

plt.figure(figsize=(7, 5))
plt.boxplot(data, labels=labels, showfliers=False)
plt.xlabel("Percentile (discriminative need)")
plt.ylabel("Distance à la courbe IB (euclidienne)")
plt.tight_layout()
plt.show()

# (Optionnel) afficher moyennes pour vérifier
for p in perc:
    print(f"Percentile {p:>2}: mean dist = {dist_nn[p].mean():.4f} | median = {np.median(dist_nn[p]):.4f}")
print(f"WCS human      : mean dist = {dist_wcs.mean():.4f} | median = {np.median(dist_wcs):.4f}")
