
from accuracy_complexite import *

results = cumpute_humain_langues("term.txt")

perc = (20, 40, 60, 80)
all_results_nn = []
for j in perc:
    results_nn = cumpute_nn_langues(f"fig3_bis/percentile{j}", 0, 60)
    all_results_nn.append(results_nn)
all_results_nn = [np.array(r) for r in all_results_nn]

betas = np.linspace(1.0, 1.15, 50)

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

print(f"\nRésultats sauvegardés dans {csv_file}")

# ---------- Plot 2D ----------
plt.figure(figsize=(7, 6))

# Langues WCS
results = np.array(results)
plt.scatter(results[:,1], results[:,2],
            s=40, alpha=0.6, label="WCS languages")
results_nn = np.array(results_nn)
# Langue NN
plt.scatter(all_results_nn[0][:, 1], all_results_nn[0][:, 2],
            s=40,
            alpha=0.6,
            color="orange",
            label="Neural language percentile 20")
plt.scatter(all_results_nn[1][:, 1], all_results_nn[1][:, 2],
            s=40,
            alpha=0.6,
            color="yellow",
            label="Neural language percentile 40")
plt.scatter(all_results_nn[2][:, 1], all_results_nn[2][:, 2],
            s=40,
            alpha=0.6,
            color="red",
            label="Neural language percentile 60")
plt.scatter(all_results_nn[3][:, 1], all_results_nn[3][:, 2],
            s=40,
            alpha=0.6,
            color="purple",
            label="Neural language percentile 80")

# Courbe IB
plt.plot(ib_points[:,0], ib_points[:,1],
         color="black", linewidth=2, label="IB curve")


# Zone interdite (au-dessus de la courbe IB)
ymax = plt.ylim()[1]
plt.fill_between(
    ib_points[:,0],
    ib_points[:,1],
    ymax,
    color="gray",
    alpha=0.25,
    hatch="///",
    label="Infeasible region"
)

plt.xlabel("Complexity, I(M;W) bits")
plt.ylabel("Accuracy, I(W;U) bits")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
