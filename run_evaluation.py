# This code is adapted from https://github.com/wbw520/DiReCT
# Credits to the original authors.

from evaluation_stats import process
import evaluate_predictions
import os, sys
from config import Qsets

os.environ["MY_MODEL"] = "llama31"

qsets = Qsets()
root = qsets.sample
# pred_name = "predicts/" + "predict_qw7b" + "_premise"
pred_name = "predicts/" + "predict_dsr1_leakexp_sample2_premise"
reverse = False
parallel = True

if len(sys.argv) > 1:
    pred_name = sys.argv[1]


if "all" in pred_name: root = qsets.all
if "sel" in pred_name: root = qsets.selected
if "sample" in pred_name: root = qsets.sample
if "sml" in pred_name: root = qsets.small

if len(sys.argv) > 2:
    root = sys.argv[2]


sep = "-" * 30
print(sep)
print("EVAL INFORMATION:")
print(sep)
print(f"{'Now evaluating:':<20} {pred_name}")
print(f"{'Judger:':<20} {os.environ.get('MY_MODEL', 'N/A')}")
print(f"{'Reverse:':<20} {reverse}")
print(f"{'Parallel:':<20} {parallel}")
print(sep)

if os.path.isdir(root) and os.path.isdir(pred_name):
    evaluate_predictions.main(root, pred_name, reverse=reverse, parallel=parallel)
    process(root=root, pred_name=pred_name + "_eval")
else:
    print("Check your path!" + pred_name)
