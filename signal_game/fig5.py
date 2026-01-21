from accuracy_complexite import *


perc = (1, 5, 10)
all_results_nn = []
for j in perc:
    results_nn = cumpute_nn_langues(f"fig5/temp{j}", 0, 60)
    all_results_nn.append(results_nn)
all_results_nn = [np.array(r) for r in all_results_nn]

results = cumpute_nn_langues(f"fig5/percentile60", 0, 60)
results = np.array(results)
rf_data = results[:,1]
data_1 = all_results_nn[0][:, 1]  # I(M;W)
data_5 = all_results_nn[1][:, 1]
data_10 = all_results_nn[2][:, 1]

plt.figure(figsize=(7, 5))
plt.boxplot([rf_data, data_1, data_5, data_10],
            tick_labels=["RF","1", "5", "10"],
            showfliers=False)


plt.ylabel("Complexity")
plt.tight_layout()
plt.show()