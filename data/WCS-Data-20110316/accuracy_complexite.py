import numpy as np
import pickle
from collections import Counter, defaultdict
import h5py
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import csv

# ---------- Chargement LAB ----------
with h5py.File("ours_images_single_sm0.h5", "r") as f:
    LAB = np.array(f["features"])  # (330, 3)

U = 330
P_c = 1.0 / U

def kl(p, q, eps=1e-12):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * np.log(p / q))

# ---------- Modèle perceptif ----------
sigma2 = 64.0 / (128**2)
D2 = cdist(LAB, LAB, metric="sqeuclidean")

m_c_u = np.exp(-D2 / (2 * sigma2))
m_c_u /= m_c_u.sum(axis=1, keepdims=True)

# p(u)
p_u = (P_c * m_c_u).sum(axis=0)

P_m = np.ones(U) / U
# ---------- Stockage des résultats ----------
results = []

# ---------- Boucle langues ----------
for LANG in range(1, 111):

    chip_terms = defaultdict(list)
    with open("term.txt", "r", encoding="utf-8") as f:
        for line in f:
            lang, speaker, chip, term = line.strip().split("\t")
            if int(lang) == LANG:
                chip_terms[int(chip) - 1].append(term)

    vocab = sorted({t for terms in chip_terms.values() for t in terms})
    word2idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)

    # ---------- P(w|c) ----------
    P_w_c = np.zeros((U, V))
    for c in range(U):
        counts = Counter(chip_terms[c])
        total = sum(counts.values())
        for w, cnt in counts.items():
            P_w_c[c, word2idx[w]] = cnt / total

    # ---------- P(w) ----------
    P_w = P_c * P_w_c.sum(axis=0)
    P_w /= P_w.sum()

    # ---------- Complexité I(M;W) ----------
    I_MW = 0.0
    for c in range(U):
        for w in range(V):
            p = P_w_c[c, w]
            if p > 0:
                I_MW += P_c * p * np.log2(p / P_w[w])

    # ---------- p(u|w) ----------
    p_u_w = np.zeros((V, U))
    for w in range(V):
        if P_w[w] > 0:
            P_c_given_w = P_w_c[:, w] * P_c / P_w[w]
            p_u_w[w] = (P_c_given_w[:, None] * m_c_u).sum(axis=0)
            p_u_w[w] /= p_u_w[w].sum()

    # ---------- Accuracy I(U;W) ----------
    I_UW = 0.0
    for w in range(V):
        if P_w[w] > 0:
            I_UW += P_w[w] * kl(p_u_w[w], p_u)

    print(f"LANG {LANG:3d} | I(M;W)={I_MW:.3f} | I(U;W)={I_UW:.3f}")

    results.append((LANG, I_MW, I_UW))
    
    
# ---------- IB Solver ----------
def solve_ib(beta, q_w_m_init, K=20, n_iter=200):
    q_w_m = q_w_m_init.copy()

    for _ in range(n_iter):
        # q(w)
        q_w = (P_m[:, None] * q_w_m).sum(axis=0)

        # q(m|w)
        q_m_w = (q_w_m * P_m[:, None]) / q_w[None, :]
        q_m_w /= q_m_w.sum(axis=0, keepdims=True)

        # \hat m_w(u)
        m_w_u = q_m_w.T @ m_c_u
        m_w_u /= m_w_u.sum(axis=1, keepdims=True)

        # KL(m || m_w)
        D = np.zeros((U, K))
        for c in range(U):
            for w in range(K):
                D[c, w] = kl(m_c_u[c], m_w_u[w])

        # mise à jour IB
        q_w_m = q_w[None, :] * np.exp(-beta * D)
        q_w_m /= q_w_m.sum(axis=1, keepdims=True)

    # === recalcul FINAL cohérent ===
    q_w = (P_m[:, None] * q_w_m).sum(axis=0)

    # Complexité I(M;W) en bits
    I_MW = 0.0
    for c in range(U):
        for w in range(K):
            p = q_w_m[c, w]
            if p > 0:
                I_MW += P_m[c] * p * np.log2(p / q_w[w])

    # Accuracy I(W;U) en bits
    p_u = (P_m[:, None] * m_c_u).sum(axis=0)
    I_UW = 0.0
    for w in range(K):
        I_UW += q_w[w] * kl(m_w_u[w], p_u) / np.log(2)

    return I_MW, I_UW, q_w_m

betas = np.linspace(1.0, 1.20, 25)

K = 20
q_w_m = np.random.rand(U, K)
q_w_m /= q_w_m.sum(axis=1, keepdims=True)

ib_points = []

for beta in betas:
    I_MW, I_UW, q_w_m = solve_ib(beta, q_w_m, K=K)
    ib_points.append((I_MW, I_UW))
    print(f"β={beta:.3f} | I(M;W)={I_MW:.3f} | I(U;W)={I_UW:.3f}")

ib_points = np.array(ib_points)

# trier par complexité
ib_points = ib_points[np.argsort(ib_points[:, 0])]




# ---------- Sauvegarde CSV ----------
csv_file = "wcs_accuracy_complexity.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Language", "Complexity_I(M;W)", "Accuracy_I(U;W)"])
    for row in results:
        writer.writerow(row)

print(f"\n✔ Résultats sauvegardés dans {csv_file}")

# ---------- Plot 2D ----------
plt.figure(figsize=(7, 6))

# Langues WCS
results = np.array(results)
plt.scatter(results[:,1], results[:,2],
            s=40, alpha=0.6, label="WCS languages")

# Courbe IB
plt.plot(ib_points[:,0], ib_points[:,1],
         color="black", linewidth=2, label="IB curve")

plt.xlabel("Complexity I(M;W) [bits]")
plt.ylabel("Accuracy I(W;U) [bits]")
plt.title("Information Bottleneck Trade-off\n(Color Naming)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
