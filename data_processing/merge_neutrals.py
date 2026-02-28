import json
import os

dataset_path = './predict_neutrals/neu05'

#############

target = "all_logits"

#############

all_preds = []

for root, dirs, files in os.walk(dataset_path):
    for filename in files:
        if "predict_" + target in filename:
            with open(os.path.join(root, filename), "r", encoding = "utf-8") as f:
                for line in f:
                    docdict = json.loads(line)
                    all_preds.append(docdict)

with open(os.path.join(dataset_path, "merged_predict_" + target + ".json"), "w", encoding = "UTF-8") as mf:
    for docdict in all_preds:
        print(json.dumps(docdict, ensure_ascii = False), file = mf)
