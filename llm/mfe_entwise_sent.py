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
import pymorphy2
from nltk.tokenize import word_tokenize
morph = pymorphy2.MorphAnalyzer()

from openai import OpenAI
from nltk.data import load
ru_tokenizer = load("tokenizers/punkt/russian.pickle")

oclient = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
print([mod.id for mod in oclient.models.list().data])

if shots_num > 0:
    all_entities = {}
    with open(train_dataset_path, "r", encoding = "UTF-8") as tf:
        for line in tf:
            line_data = json.loads(line)
            line_text = line_data["text"]
            entities = list(zip(line_data["entity_start_chars"], line_data["entity_end_chars"], line_data["entity_types"]))
            sentence_spans = ru_tokenizer.span_tokenize(line_text)
            for start, end in sentence_spans:
                curr_entities = [(s, e, t) for s, e, t in entities if start <= s and e <= end]
                for s, e, t in curr_entities:
                    if t not in all_entities:
                        all_entities[t] = [line_text[s : e]]
                    else:
                        all_entities[t].append(line_text[s : e])

    comp2ent = {}
    all_comps = {}
    for t in all_entities.keys():
        all_comps[t] = []
        for entity in all_entities[t]:
            words = ",".join(sorted([morph.parse(word)[0].normal_form for word in word_tokenize(entity)]))
            if words not in comp2ent:
                comp2ent[words] = entity
            all_comps[t].append(words)
        all_comps[t] = Counter(all_comps[t]).most_common(shots_num)

    # print(list(all_entities.items()))

    mfe_prompt = "\nHere are most frequent examples for each entity class, separated by comma:\n" + "\n".join([t + ": " + ", ".join([comp2ent[e[0]] for e in es]) for t, es in sorted(list(all_comps.items()), key = lambda x : x[0])])

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
                    "answer" : {line_text[s : e] : t for s, e, t in curr_entities}
                })

    chosen_shots = random.sample(shots, shots_num)
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

# with open(desc_path, "r", encoding = "UTF-8") as df:
#     desc_text = df.read()

desc_text = ""

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
        line_text = line_data["text"]
        sentence_spans = ru_tokenizer.span_tokenize(line_text)
        for s_idx, (start, end) in enumerate(sentence_spans):
            text = line_text[start : end]
            myprompt = prompt_base + mfe_prompt + shots_prompt + f"""
###
Text: {text}
Answer: """
  
            # print(myprompt)
            # exit(1)
            
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
                all_res.append((line_data["id"], s_idx, start, ans_dict))
                # print((line_data["id"], s_idx, start, ans_dict))

            except:
                print("Could not retrieve json dict:")
                print((start, res.choices[0].message.content))
                other_res.append((line_data["id"], s_idx, start, res.choices[0].message.content))

os.makedirs("res", exist_ok=True)

with open("res/mfe-entwise-sent-seed=" + str(seed) + "-" + model.replace('/','-') + "-" + tags_path.split('/')[-1].replace('.tags','') + "-" + str(shots_num) + "shot.jsonl", "w", encoding = "UTF-8") as af:
    for res in all_res:
        af.write(json.dumps(res, ensure_ascii = False) + "\n")

with open("res/mfe-entwise-sent-seed=" + str(seed) + "-" + model.replace('/','-') + "-" + tags_path.split('/')[-1].replace('.tags','') + "-" + str(shots_num) + "shot-other.jsonl", "w", encoding = "UTF-8") as af:
    for res in other_res:
        line_data["id"], s_idx, start, content = res
        af.write(json.dumps((line_data["id"], s_idx, start), ensure_ascii = False) + "|||||||||||" + content + "\n")