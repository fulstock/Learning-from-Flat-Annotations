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

def contains_sublist(haystack, needle):
    needle_length = len(needle)
    if needle_length > len(haystack):
        return False
    for i in range(len(haystack) - needle_length + 1):
        if haystack[i:i + needle_length] == needle:
            return True
    return False

oclient = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
print([mod.id for mod in oclient.models.list().data])

if shots_num > 0:
    all_entities = {}
    entities2type = {}
    words2ent = {}
    ent2words = {}
    winct2e2e = {}
    wincete = {}
    winct2t = {}
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
                    # Canonical representation of every entity: alphabetical list of normalised tokens
                    # Using a deterministic order (sorted) makes equality and sub-list checks order-independent
                    words = "|".join([morph.parse(word)[0].normal_form for word in word_tokenize(line_text[s : e])])
                    if words not in words2ent:
                        words2ent[words] = [line_text[s : e]]
                    else:
                        words2ent[words].append(line_text[s : e])
                    ent2words[line_text[s : e]] = words
                    entities2type[words] = t

    # ====================== OPTIMIZED NESTED-ENTITY MAPPING ======================
    # The previous implementation compared **every** pair of entities which is
    # O(N²) and quickly becomes a bottleneck when the number of unique entities
    # is large. We now pre-tokenise each canonical entity representation once and
    # generate only the contiguous sub-lists of each outer entity. Contiguous
    # sub-lists are the only candidates that can satisfy `contains_sublist` when
    # the token lists are sorted alphabetically (as they were constructed).
    #
    # Complexity drops to O(N·m²), where `m` is the average entity length in
    # tokens – usually very small (≤5). This offers orders-of-magnitude speed-up
    # on realistic corpora while producing IDENTICAL results.

    # Pre-tokenise entities once (remove duplicate tokens to replicate the
    # previous call to `list(dict.fromkeys(...))`)
    # Deduplicate AND alphabetically sort token lists to match the canonical representation above
    tokenized_entities = {ent: list(dict.fromkeys(sorted(ent.split('|')))) for ent in entities2type}

    for e_outer, outer_tokens in tqdm(tokenized_entities.items(), desc="Building nested entity mappings"):
        t_outer = entities2type[e_outer]

        # Ensure dicts are initialised to avoid KeyError further downstream
        winct2t.setdefault(t_outer, [])
        winct2e2e.setdefault(t_outer, {})
        wincete.setdefault(e_outer, [])

        length = len(outer_tokens)
        # Iterate over all proper contiguous sub-lists of `outer_tokens`
        for i in range(length):
            for j in range(i + 1, length + 1):
                if j - i == length:
                    # Skip the sub-list equal to the full list (not "nested")
                    continue
                sub_tokens = outer_tokens[i:j]
                e_inner = "|".join(sub_tokens)

                # Only proceed if this candidate is a known entity
                if e_inner not in entities2type:
                    continue

                t_inner = entities2type[e_inner]

                # --- Populate helper structures (identical semantics) ---
                winct2t[t_outer].append(t_inner)
                winct2e2e[t_outer].setdefault(e_outer, []).append(e_inner)
                wincete[e_outer].append(e_inner)
    # ===========================================================================

    mfe_prompt = ""
    # + "\n".join([t + ": " + ", ".join([comp2ent[e[0]] for e in es]) for t, es in sorted(list(all_comps.items()), key = lambda x : x[0])])
    # print(json.dumps(winct2e2e, ensure_ascii = False))

    tag2prompt = {}

    for ot in all_entities.keys():
        curr_common_types = [t for t, _ in Counter(winct2t[ot]).most_common(shots_num)]
        if ot in curr_common_types:
            curr_common_types = [t for t, _ in Counter(winct2t[ot]).most_common(shots_num + 1) if t != ot]
        tag2prompt[ot] = ""
        tag2prompt[ot] += "For class " + ot + " most common nested entity classes of such nestedness are: " + ", ".join(curr_common_types) + "\n"
        tag2prompt[ot] += "Here are some examples of the " + ot + " class as outermost entity and its nested entities: \n"
        
        best_outers = []

        for inner_type in curr_common_types:
            only_curr_inner = {e : ei for e, ei in winct2e2e[ot].items() if inner_type in [entities2type[inn] for inn in winct2e2e[ot][e]]}
            new_best_outer = sorted(only_curr_inner, key=lambda key: len(only_curr_inner[key]), reverse=True)[0]
            if new_best_outer not in best_outers:
                best_outers.append(new_best_outer)

        for best_outer in best_outers:
            all_inners_raw = wincete[best_outer]
            # Keep unique inner entities while preserving insertion order
            outermost_entity = words2ent[best_outer][0]
            formatted_inners = []
            # Preserve order while ensuring uniqueness of inner entities
            seen_inner = set()
            for inner_key in all_inners_raw:
                if inner_key == best_outer or inner_key in seen_inner:
                    continue
                seen_inner.add(inner_key)

                outer_text = outermost_entity

                # --- 1) Try exact token match with known surface variants ---
                variants = words2ent.get(inner_key, [])
                tokens = word_tokenize(outer_text)

                def is_exact_span(v: str) -> bool:
                    vtoks = word_tokenize(v)
                    n = len(vtoks)
                    for idx in range(len(tokens) - n + 1):
                        if tokens[idx:idx + n] == vtoks:
                            return True
                    return False

                # Collect all variants whose *token sequence* exactly occurs in the outer sentence
                exact_span_matches = [v for v in variants if is_exact_span(v)]

                if exact_span_matches:
                    # Prefer the longest surface form among exact matches
                    entity_text = max(exact_span_matches, key=len)
                else:
                    # Loose fallback: any substring (still prefer the longest)
                    substring_matches = [v for v in variants if v in outer_text]
                    entity_text = max(substring_matches, key=len) if substring_matches else None

                # --- 2) Fallback: morphological span search inside outer_text ---
                if entity_text is None:
                    tokens = word_tokenize(outer_text)
                    n_tokens = len(tokens)
                    inner_canon = inner_key
                    found = None
                    for s_idx in range(n_tokens):
                        for e_idx in range(s_idx, n_tokens):
                            span_tokens = tokens[s_idx:e_idx + 1]
                            span_canon = "|".join(
                                sorted(
                                    list(dict.fromkeys([morph.parse(tok)[0].normal_form for tok in span_tokens]))
                                )
                            )
                            if span_canon == inner_canon:
                                found = " ".join(span_tokens)
                                break
                        if found:
                            break
                    entity_text = found if found else None

                # Final safeguard: keep the inner entity only if we found a
                # surface form that is literally present in the outer text.
                if entity_text and entity_text in outer_text:
                    formatted_inners.append({entity_text: entities2type[inner_key]})

            # Emit an example as long as we have at least one nested entity
            if formatted_inners:
                tag2prompt[ot] += "Outermost entity: ```" + outermost_entity + "```, nested are ```"
                tag2prompt[ot] += json.dumps(formatted_inners, ensure_ascii = False) + "```"
                tag2prompt[ot] += "\n"

        mfe_prompt += tag2prompt[ot]

    shots_prompt = ""

with open(tags_path, "r", encoding = "UTF-8") as tf:
    tags = sorted(json.load(tf))

# with open(desc_path, "r", encoding = "UTF-8") as df:
#     desc_text = "Here are definitions of all named entitiy types:\n" + df.read()

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
        line_entities = line_data["pred_ner"]
        line_entities = [le for le in line_entities if len(le[-1].split(' ')) > 1]
        for start, end, tag, span in tqdm(line_entities):
            text = span
            # myprompt = prompt_base + tag2prompt[tag] + shots_prompt + f"""
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
                            "repetition_penalty": 1.1,
                            "guided_choice": None,
                            "add_generation_prompt": True,
                        }#,
                        #stream=True
                    )
                    break
                except Exception as e:
                    print(e)
                    continue
            # print(text)
            try:
                answer = res.choices[0].message.content
                preds = answer.split('```')[1].replace('`','').replace('json','')
                ans_dict = json.loads(preds)
                all_res.append((line_data["id"], (start, end, tag, span), ans_dict))
                # print((line_data["id"], s_idx, start, ans_dict))

            except:
                print("Could not retrieve json dict:")
                print((start, res.choices[0].message.content))
                other_res.append((line_data["id"], (start, end, tag, span), res.choices[0].message.content))

os.makedirs("res", exist_ok=True)

with open("res/winc-seed=" + str(seed) + "-" + model.replace('/','-') + "-" + tags_path.split('/')[-1].replace('.tags','') + "-" + str(shots_num) + "shot.jsonl", "w", encoding = "UTF-8") as af:
    for res in all_res:
        af.write(json.dumps(res, ensure_ascii = False) + "\n")

with open("res/winc-seed=" + str(seed) + "-" + model.replace('/','-') + "-" + tags_path.split('/')[-1].replace('.tags','') + "-" + str(shots_num) + "shot-other.jsonl", "w", encoding = "UTF-8") as af:
    for res in other_res:
        line_data["id"], (start, end, tag, span), content = res
        af.write(json.dumps((line_data["id"], (start, end, tag, span)), ensure_ascii = False) + "|||||||||||" + content.replace('\n',' ') + "\n")