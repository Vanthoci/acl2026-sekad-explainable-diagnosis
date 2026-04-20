# This code is adapted from https://github.com/wbw520/DiReCT
# Credits to the original authors.

from typing import List, Optional
import json
import os
import sys
from difflib import SequenceMatcher

import concurrent.futures
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import fire

from utils.data_analysis import (
    cal_a_json,
    get_all_file_paths,
    deduction_assemble,
    capitalize_first_letter
)
from utils.data_extraction import (
    discriminate_similarity_observation,
    discriminate_similarity_reason
)
from utils.gpt_call import one_contact


def main(
    root: str,
    pred_name: str,
    temperature: float = 0,
    top_p: float = 1,
    max_seq_len: int = 8192,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    reverse=False,
    parallel: bool = False,
):
    all_files_gt = get_all_file_paths(root)
    all_files_pred = get_all_file_paths(pred_name)
    all_files_pred_eval = get_all_file_paths(pred_name + "_eval")
    if reverse:
        all_files_gt = all_files_gt[::-1]
        all_files_pred = all_files_pred[::-1]
        all_files_pred_eval = all_files_pred_eval[::-1]

    def process_file(i):
        root_file = all_files_gt[i]
        root_pred = root_file.replace(root, pred_name)
        root_eval = root_file.replace(root, pred_name + "_eval")
        if root_eval in all_files_pred_eval:
            return
        if root_pred not in all_files_pred:
            return
        try:
            deal_a_file(root_file, root_pred, root_eval, None, max_gen_len, temperature, top_p)
        except Exception as e:
            print('ERROR', e, root_eval)

    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            list(tqdm(executor.map(process_file, range(len(all_files_gt))), total=len(all_files_gt)))
    else:
        for i in tqdm(range(len(all_files_gt))):
            process_file(i)


def deal_a_file(root_file, root_pred, root_eval, generator, max_gen_len, temperature, top_p):
    record_node, input_content, chain_gt = cal_a_json(root_file)
    GT = deduction_assemble(record_node)

    with open(root_pred, 'r') as file:
        predict = json.load(file)

    chain_pred = predict.pop("chain")

    GT_observation = list(GT.keys())
    predict_observation = list(predict.keys())
    len_ob_gt = len(GT_observation)
    len_ob_pred = len(predict_observation)

    def normalized_lcs(s1: str, s2: str) -> float:
        lcs_len = SequenceMatcher(None, s1.lower(), s2.lower()).find_longest_match().size
        return lcs_len / min(len(s1), len(s2)) if min(len(s1), len(s2)) > 0 else 0.0

    similarity_matrix = np.zeros((len_ob_gt, len_ob_pred))
    for i, gt_ob in enumerate(GT_observation):
        for j, pred_ob in enumerate(predict_observation):
            similarity_matrix[i, j] = normalized_lcs(gt_ob, pred_ob)

    row_ind, col_ind = linear_sum_assignment(similarity_matrix, maximize=True)
    ob_record = list(zip(row_ind, col_ind))

    record = {
        "chain_gt": chain_gt,
        "chain_pred": chain_pred,
        "len_ob_gt": len_ob_gt,
        "len_ob_pred": len_ob_pred,
        "ob_record_paired": {}
    }

    for item in ob_record:
        i, j = item
        if similarity_matrix[i, j] < 0.5:
            continue
        disease_gt = GT[GT_observation[i]][-1]
        disease_pred = predict[predict_observation[j]][-1]
        re_gt = GT[GT_observation[i]][0]
        re_pred = predict[predict_observation[j]][0]
        input_reason = discriminate_similarity_reason(re_gt, re_pred)
        result_reason = one_contact(input_reason, log_prefix="eval")
        record["ob_record_paired"].update({str(item): [disease_gt, disease_pred, re_gt, re_pred, result_reason]})

    directory = os.path.dirname(root_eval)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(root_eval, 'w') as json_file:
        json.dump(record, json_file)


if __name__ == "__main__":
    fire.Fire(main)
