from accuracy_complexite import *

results = cumpute_humain_langues("term.txt")
results = np.array(results)
perc = (20, 40, 60, 80)
all_results_nn = []
for j in perc:
    results_nn = cumpute_nn_langues(f"fig4/percentile{j}", 0, 60)
    all_results_nn.append(results_nn)
all_results_nn = [np.array(r) for r in all_results_nn]

h_data = results[:,1]
data_20 = all_results_nn[0][:, 1]  # I(M;W)
data_40 = all_results_nn[1][:, 1]
data_60 = all_results_nn[2][:, 1]
data_80 = all_results_nn[3][:, 1]

plt.figure(figsize=(7, 5))
plt.boxplot([data_20, data_40, data_60, data_80, h_data],
            labels=["20", "40", "60", "80", "Human systems"],
            showfliers=False)

plt.xlabel("Percentile")
plt.ylabel("Complexity")
plt.tight_layout()
plt.show()