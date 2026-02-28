import os
import json
import argparse

parser = argparse.ArgumentParser(description="Prepare 5-fold cross-validation data for corruption.")
parser.add_argument('--original_data', type=str, required=True, help="Path to original flat train.json")
parser.add_argument('--damaged_data', type=str, required=True, help="Path to damaged train.json")
parser.add_argument('--output_dir', type=str, required=True, help="Path to output parted data directory")
parser.add_argument('--total_parts', type=int, default=5)
args = parser.parse_args()

total_parts = args.total_parts

original_data = args.original_data
damaged_data = args.damaged_data
parted_data = args.output_dir

with open(original_data, "r", encoding = "UTF-8") as origfile:
    origlines = origfile.readlines()
with open(damaged_data, "r", encoding = "UTF-8") as dmgfile:
    dmglines = dmgfile.readlines()

part_delims = [int(len(origlines) * part / total_parts) for part in range(total_parts)]
part_delims.append(len(origlines))
part_paired_delims = [(p1, p2 - 1) for p1, p2 in zip(part_delims, part_delims[1:])]

for part in range(total_parts):
    train_lines = [d for d_idx, d in enumerate(dmglines) if d_idx < part_paired_delims[part][0] or d_idx > part_paired_delims[part][1]]
    predict_lines = [o for o_idx, o in enumerate(origlines) if o_idx >= part_paired_delims[part][0] and o_idx <= part_paired_delims[part][1]]
    if not os.path.exists(parted_data + "/part" + str(part + 1)):
        os.makedirs(parted_data + "/part" + str(part + 1))
    with open(parted_data + "/part" + str(part + 1) + "/train.json", "w", encoding = "UTF-8") as partfile:
        partfile.writelines(train_lines)
    with open(parted_data + "/part" + str(part + 1) + "/dev.json", "w", encoding = "UTF-8") as partfile:
        partfile.writelines(predict_lines)