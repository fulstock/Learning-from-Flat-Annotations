# Learning Nested Named Entity Recognition from Flat Annotations

## Introduction

This is the repository for our "Learning Nested Named Entity Recognition from Flat Annotations" paper, accepted at EACL 2026 Student Research Workshop.

In this paper, we investigate whether models can learn nested entity structure from flat annotations alone. We evaluate four approaches on the NEREL dataset (29 entity types, Russian): string inclusions (substring matching), entity corruption (pseudo-nested data via cross-prediction), flat neutralization (reducing false negative signal), and a hybrid fine-tuned + LLM pipeline. Our best combined method achieves 26.37% inner F1, closing 40% of the gap to full nested supervision.

## Binder model

Binder model (with our fixes and additional featured scripts) used in our study is included in the `binder/` directory. It is based on the original [PrincetonNLP/binder](https://github.com/PrincetonNLP/binder) bi-encoder model. All Binder experiments use `binder/run_ner.py` with JSON configuration files:

```bash
python binder/run_ner.py config.json
```

The JSON config specifies model paths, data paths, entity type file, hyperparameters, and training arguments. See the [HuggingFace TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) documentation for available options. For Binder training, download [RuRoBERTa-large](https://huggingface.co/ai-forever/ruRoberta-large) or another XLM-RoBERTa variant as the base language model.

## NEREL dataset

We use the [NEREL v1.1](https://github.com/nerel-ds/NEREL) dataset -- a Russian nested NER benchmark with 29 entity types. Download it from [https://github.com/nerel-ds/NEREL](https://github.com/nerel-ds/NEREL) and place it in a directory (e.g., `./NEREL/`).

Entity type definitions used in LLM experiments can be found at `data/entity_definitions.txt`, and the tag list at `data/nerel.tags`.

## Scripts

All scripts were run on Python 3.10. You can find needed packages at `requirements.txt` and install them via `pip install -r requirements.txt`. Additionally, install `torch`, `transformers`, and `datasets` for Binder training. NLTK tokenizer data is also required:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

### Data Preparation

- `binder/data_preproc/brat_to_hfds.py`: convert NEREL BRAT annotations to Binder JSONL format. Creates `train.json`, `dev.json`, `test.json`.
- `data_processing/make_spans.py`: convert BRAT annotations to span-level format.

```bash
python binder/data_preproc/brat_to_hfds.py \
    --brat_dataset_path ./NEREL \
    --tags_path ./data/nerel.tags \
    --hfds_output_path ./data/NEREL-binder
```

### Nested NER from Flat Supervision Approaches

#### Inclusions and combined approaches

- `data_processing/prepare_lemwincdamage_dataset.py`: combine *lemmatized inclusions* with *damaged cross-prediction* data. Corresponds to the combined *lemwinc+dmg* approaches described in the paper. Damaged cross-predicted data should be prepared before running this script (see below).

```bash
python data_processing/prepare_lemwincdamage_dataset.py \
    --data_dir ./data/NEREL-binder \
    --train_file train.json --dev_file dev.json --test_file test.json \
    --damage_data_file ./converted/end-diglets.json \
    --data_format nerel \
    --output_dir ./data/NEREL-lemwinc-dmg-binder
```

#### Entity corruption (damaged cross-prediction)

Should be run in the following order:

1. `data_processing/damage_data.py`: corrupt words in long entities to create pseudo-nested data;
2. `data_processing/data4cross_predict.py`: prepare 5-fold cross-validation splits from corrupted data;
3. Train Binder on each fold, predict on held-out part (run `binder/run_ner.py` for each fold);
4. `data_processing/convert_and_merge.py`: merge fold predictions with flat data for final training.

```bash
# 1. Generate corrupted data
python data_processing/damage_data.py \
    --input_dir ./data/NEREL-outerflat-binder \
    --output_dir ./data/NEREL-outerflat-dmg-binder \
    --method end --mask diglets

# 2. Create cross-validation splits
python data_processing/data4cross_predict.py \
    --original_data ./data/NEREL-outerflat-binder/train.json \
    --damaged_data ./data/NEREL-outerflat-dmg-binder/end/diglets/train.json \
    --output_dir ./data/NEREL-outerflat-parted-binder/end/diglets

# 3. Train Binder on each fold, predict on held-out part
#    (run binder/run_ner.py for each fold)

# 4. Merge predictions
python data_processing/convert_and_merge.py \
    --predicted_parts_path ./predicted_parts/end/diglets \
    --original_data_path ./data/NEREL-outerflat-binder/train.json \
    --output_path ./converted/end-diglets.json
```

#### Flat neutralization

- `data_processing/convert_neutrals.py`: convert Binder neutral predictions format;
- `data_processing/merge_neutrals.py`: merge neutral predictions for combined training.

These scripts operate on prediction outputs from `./predict_neutrals/` directory.

### LLM Approaches

All LLM scripts require an OpenAI-compatible API server and share the same interface:

```bash
python llm/<script>.py \
    --train_dataset_path ./data/NEREL-binder/train.jsonl \
    --tags_path ./data/nerel.tags \
    --desc_path ./data/entity_definitions.txt \
    --test_dataset_path ./data/NEREL-binder/test.jsonl \
    --seed 33 --shots_num 5 \
    --model deepseek-r1-32b
```

Run each experiment with seeds 33, 55, 77 for statistical validity.

#### Pure LLM

- `llm/normal.py`: baseline with random sentence selection for few-shot examples;
- `llm/mfe.py`: *Most Frequent Entity* selection -- selects examples containing the most frequently occurring entities;
- `llm/mfe_entwise.py`: *entity-wise* selection -- top N examples for each of 29 entity classes;
- `llm/mfe_entwise_sent.py`: sentence-level entity-wise selection with morphological normalization.

#### Hybrid Binder+LLM

- `llm/winc.py`: Binder detects outer entities, LLM identifies nested entities within each span;
- `llm/winc_small.py`: same as `winc.py`, but with entity-type-specific prompts.

For hybrid approaches, the test file must include a `pred_ner` field with Binder predictions.

### Evaluation

- `evaluation/check_f1_nested.py`: compute micro and macro F1 scores for overall, inner, and outer entities.

```bash
# Place prediction files in ./preds/ directory
# Each file: JSONL with "gold_ner" and "pred_ner" fields
python evaluation/check_f1_nested.py
```

## Contacts

For any inquiries you can reach out to [fulstocky@gmail.com](mailto:fulstocky@gmail.com).
