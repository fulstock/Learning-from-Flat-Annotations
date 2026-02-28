import requests
from tqdm.auto import tqdm
import os
import random
import json
import argparse

brat2mrc_parser = argparse.ArgumentParser(description = "Brat to hfds-json formatter script.")
brat2mrc_parser.add_argument('--train_dataset_path', type = str, required = True)
brat2mrc_parser.add_argument('--tags_path', type = str, required = True)
brat2mrc_parser.add_argument('--desc_path', type = str, required = True)
brat2mrc_parser.add_argument('--test_dataset_path', type = str, required = True)
brat2mrc_parser.add_argument('--seed', type = int, required = True)
brat2mrc_parser.add_argument('--shots_num', type = int, required = True)
brat2mrc_parser.add_argument('--model', type = str, required = True)

args = brat2mrc_parser.parse_args()

# ===============

train_dataset_path = args.train_dataset_path
tags_path = args.tags_path
desc_path = args.desc_path
test_dataset_path = args.test_dataset_path
seed = args.seed
shots_num = args.shots_num
model = args.model

# ===============

random.seed(seed)

import json
from collections import Counter

from openai import OpenAI
from nltk.data import load
ru_tokenizer = load("tokenizers/punkt/russian.pickle")

oclient = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
print([mod.id for mod in oclient.models.list().data])

if shots_num > 0:
    shots = []
    with open(train_dataset_path, "r", encoding = "UTF-8") as tf:
        for line in tf:
            line_data = json.loads(line)
            line_text = line_data["text"]
            entities = list(zip(line_data["entity_start_chars"], line_data["entity_end_chars"], line_data["entity_types"]))
            sentence_spans = ru_tokenizer.span_tokenize(line_text)
            for start, end in sentence_spans:
                curr_entities = [(s, e, t) for s, e, t in entities if start <= s and e <= end]
                shots.append({
                    "text" : line_text[start : end],
                    "entities" : [line_text[s : e] for s, e, _ in curr_entities],
                    "answer" : {line_text[s : e] : t for s, e, t in curr_entities}
                })

    spans = [e for s in shots for e in s["entities"]]
    span_to_contexts = {}
    for shot in shots:
        for e in shot["entities"]:
            if e in span_to_contexts:
                span_to_contexts[e].append(shot)
            else:
                span_to_contexts[e] = [shot]
    span_count = Counter(spans)
    # print([(s, [span_count[e] for e in s["entities"]]) for s in shots[:1]])
    most_common_entities = [e for e, _ in span_count.most_common(shots_num)]
    ranked_shots = [random.choice(span_to_contexts[e]) for e in most_common_entities]
    # print(ranked_shots)

    chosen_shots = ranked_shots
    shots_prompt = """
Here are examples with desired format of output.
###
""" + """
###
###
""".join([f"""Text {idx + 1}: {chosen_shots[idx]['text']}
Answer {idx + 1}: ```{json.dumps(chosen_shots[idx]["answer"], ensure_ascii = False)}```""" for idx in range(len(chosen_shots))]) + """
###"""
else:
    shots_prompt = ""

with open(tags_path, "r", encoding = "UTF-8") as tf:
    tags = sorted(json.load(tf))

with open(desc_path, "r", encoding = "UTF-8") as df:
    desc_text = df.read()

prompt_base = """Given entity label set: """ + str(tags) + """.
You are an excellent linguist and annotator. Based on the given entity label set, please recognize the named entities in the given text. Consider there might be a nested case, where one entity contains another. There are two possible type of nested entities:
NDT: It consists of an entity containing a shorter entity tagged with a different type.
NST: This case usually occurs when entities are originally represented by a hierarchy.
Give me ONLY entities in format of a json dictionary with named entities as keys and their types as values like this: {entity : type}. Do not write any additional text. Enclose answer in ```.
""" + desc_text

all_res = []
other_res = []

with open(test_dataset_path, "r", encoding = "UTF-8") as tf:
    for line in tqdm(tf.readlines()):
        line_data = json.loads(line)
        text = line_data["text"]
        myprompt = prompt_base + shots_prompt + f"""
###
Text: {text}
Answer: """

        
                
        while True:
            try:
                res = oclient.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": myprompt}],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=5000,
                    extra_body= {
                        "repetition_penalty": 1.05,
                        "guided_choice": None,
                        "add_generation_prompt": True,
                    }#,
                    #stream=True
                )
                break
            except:
                continue
        # print(text)
        try:
            answer = res.choices[0].message.content
            preds = answer.split('```')[1].replace('`','').replace('json','')
            ans_dict = json.loads(preds)
            all_res.append(ans_dict)
            # print(ans_dict)

        except:
            print("Could not retrieve json dict:")
            print(res.choices[0].message.content)
            other_res.append(res.choices[0].message.content)
                    

os.makedirs("res", exist_ok=True)

with open("res/mfe-seed=" + str(seed) + "-" + model.replace('/','-') + "-" + tags_path.split('/')[-1].replace('.tags','') + "-" + str(shots_num) + "shot.jsonl", "w", encoding = "UTF-8") as af:
    for res in all_res:
        af.write(json.dumps(res, ensure_ascii = False) + "\n")

with open("res/mfe-seed=" + str(seed) + "-" + model.replace('/','-') + "-" + tags_path.split('/')[-1].replace('.tags','') + "-" + str(shots_num) + "shot-other.jsonl", "w", encoding = "UTF-8") as af:
    for res in other_res:
        af.write(json.dumps(res, ensure_ascii = False) + "\n")