import os
import json
import argparse

parser = argparse.ArgumentParser(description="Merge fold predictions after corruption.")
parser.add_argument('--predicted_parts_path', type=str, required=True, help="Path to predicted parts directory")
parser.add_argument('--original_data_path', type=str, required=True, help="Path to original flat train.json")
parser.add_argument('--output_path', type=str, required=True, help="Path to output merged file")
args = parser.parse_args()

predicted_parts_path = args.predicted_parts_path
original_data_path = args.original_data_path
converted_merged_path = args.output_path

predicted_docs = []
original_docs = []

with open(original_data_path, "r", encoding = "UTF-8") as origfile:
    for line in origfile:
        original_docs.append(json.loads(line))

for root, dirs, files in os.walk(predicted_parts_path):
    for filename in files:
        with open(os.path.join(root, filename), "r") as f:
            for line in f:
                docdict = json.loads(line)
                predicted_docs.append(docdict)
                # docdict["text"] = docdict["text"].decode('utf-8')

assert len(predicted_docs) == len(original_docs)

new_docs = []
for orig_doc in original_docs:
    orig_id = orig_doc["id"]
    pred_doc = [d for d in predicted_docs if d["id"] == orig_id][0]

    pred_start_chars, pred_end_chars, pred_types, _ = [list(l) for l in zip(*pred_doc["pred_ner"])]
    orig_start_chars, orig_end_chars, orig_types = orig_doc["entity_start_chars"], orig_doc["entity_end_chars"], orig_doc["entity_types"]

    # validate errors

    entity_dict = {}
    preds = list(zip(pred_start_chars, pred_end_chars, pred_types))
    origs = list(zip(orig_start_chars, orig_end_chars, orig_types))

    for o_s, o_e, o_t in origs:
        entity_dict[f"{o_s},{o_e}"] = o_t
    for p_s, p_e, p_t in preds:
        if f"{p_s},{p_e}" not in entity_dict.keys():
            entity_dict[f"{p_s},{p_e}"] = p_t

    new_entities = []
    for key, value in entity_dict.items():
        s, e = key.split(',')
        t = value 
        new_entities.append((int(s), int(e), t))

    new_entities = sorted(list(new_entities), key = lambda x : x[0])

    new_start_chars, new_end_chars, new_types = [list(l) for l in zip(*new_entities)]

    new_doc = {
        "text" : orig_doc["text"],
        "entity_types" : new_types,
        "entity_start_chars" : new_start_chars,
        "entity_end_chars" : new_end_chars,
        "id" : orig_id,
        "word_start_chars" : orig_doc["word_start_chars"],
        "word_end_chars" : orig_doc["word_end_chars"]
    }
    new_docs.append(new_doc)

with open(converted_merged_path, "w", encoding = "utf-8", newline = "") as out:
    for doc in new_docs:
        print(json.dumps(doc, ensure_ascii = False), file = out)
        