import numpy as np
from collections import defaultdict, Counter
import pickle

LANG = 1  # langue choisie

for i in range(1, 111):

    chip_terms = defaultdict(list)

    with open("term.txt", "r", encoding="utf-8") as f:
        for line in f:
            lang, speaker, chip, term = line.strip().split("\t")
            if int(lang) == LANG:
                chip_terms[int(chip)].append(term)

    labels = []

    for chip_id in range(1, 331):
        terms = chip_terms[chip_id]
        majority_term = Counter(terms).most_common(1)[0][0]
        labels.append(majority_term)

    labels = np.array(labels)

    with open(f"ours_images_single_sm{i}.objects", "wb") as f:
        pickle.dump(labels, f)
