import json
import os
from pathlib import Path

for root, dirs, files in os.walk('./predict_neutrals'):
    for filename in files:
        if filename in ["predict_neutrals.json", "predict_predictions.json"]:
            with open(os.path.join(root, filename), "r") as f:
                with open(os.path.join(root, "conv_preds_" + filename.split('_')[-1].split('.')[0] + ".json"), "w", encoding = "utf-8", newline = "") as out:
                    for line in f:
                        docdict = json.loads(line)
                        # docdict["text"] = docdict["text"].decode('utf-8')
                        print(json.dumps(docdict, ensure_ascii = False), file = out)
            p = Path(os.path.join(root, filename))
            p.unlink()
