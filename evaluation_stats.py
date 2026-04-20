# This code is adapted from https://github.com/wbw520/DiReCT
# Credits to the original authors.

import json
import numpy as np
from transformers import BertTokenizer
from utils.data_analysis import cal_a_json, get_all_file_paths, prepare_note, disease_category, extract_keys, get_non_dict_keys, deduction_assemble, capitalize_first_letter


domain = {"Cardiology": ["Acute Coronary Syndrome", "Aortic Dissection", "Atrial Fibrillation", "Cardiomyopathy", "Heart Failure", "Hyperlipidemia", "Hypertension"],
          "Gastroenterology": ["Gastritis", "Gastro-oesophageal Reflux Disease", "Peptic Ulcer Disease", "Upper Gastrointestinal Bleeding"],
          "Neurology ": ["Epilepsy", "Migraine", "Multiple Sclerosis", "Stroke", "Alzheimer"],
          "Pulmonology ": ["Asthma", "COPD", "Pneumonia", "Pulmonary Embolism", "Tuberculosis"],
          "Endocrinology ": ["Diabetes", "Pituitary Disease", "Thyroid Disease", "Adrenal Insufficiency"]}


def process(root, pred_name, verbose=False):
    all_files_gt = get_all_file_paths(root)
    all_files_pred_eval = get_all_file_paths(pred_name)
    disease_options, flowchart = disease_category()
    record = {}
    for item in disease_options:
        record.update({item: {"acc_cat": [], "acc_diag": [], "comp_pre": [], "comp_re": [], "comp_coverage": [], "faith_ob": [], "faith_all": []}})

    min_comp_coverage = float('inf')
    min_coverage_file = None
    min_coverage_scores = {}
    for i in range(len(all_files_gt)):
        # print(f"{i}/{len(all_files_gt)}")
        root_file = all_files_gt[i]
        current_disease = root_file.split("/")[1]
        root_eval = root_file.replace(root, pred_name)
        # print(root_eval)

        if root_eval not in all_files_pred_eval:
            # acc_cat, acc_diag, comp_pre, comp_re, comp_coverage, faith_ob, faith_all = 0, 0, 0, 0, 0, 0, 0
            continue
        else:
            acc_cat, acc_diag, comp_pre, comp_re, comp_coverage, faith_ob, faith_all = statistic_one_pred(root_eval)

        if root_eval in all_files_pred_eval:
            acc_cat, acc_diag, comp_pre, comp_re, comp_coverage, faith_ob, faith_all = statistic_one_pred(root_eval)

            scores = [acc_cat, acc_diag, comp_pre, comp_re, comp_coverage, faith_ob, faith_all]
            score_names = ["acc_cat", "acc_diag", "comp_pre", "comp_re", "comp_coverage", "faith_ob", "faith_all"]

            for name, value in zip(score_names, scores):
                record[current_disease][name].append(value)

            if comp_coverage < min_comp_coverage and acc_diag == 1:
                min_comp_coverage = comp_coverage
                min_coverage_file = root_file
                min_coverage_scores = dict(zip(score_names, scores))

    # if min_coverage_file:
    #     print("\n--- Lowest comp_coverage File ---")
    #     print(f"Root File: {min_coverage_file}")
    #     print("Scores:")
    #     for score_name, score_value in min_coverage_scores.items():
    #         print(f"  {score_name}: {score_value}")
    # else:
    #     print("\nNo valid prediction files found to calculate comp_coverage.")
        
        if verbose:
            print("acc_cat:", acc_cat)
            print("acc_diag:", acc_diag)
            print("comp_pre:", comp_pre)
            print("comp_re:", comp_re)
            print("comp_coverage:", comp_coverage)
            print("faith_ob:", faith_ob)
            print("faith_all:", faith_all)

    record_acc_cat_all = []
    record_acc_diag_all = []
    record_comp_pre_all = []
    record_comp_re_all = []
    record_comp_coverage_all = []
    record_faith_ob_all = []
    record_faith_all_all = []
    
    for key, value in domain.items():
        record_acc_cat = []
        record_acc_diag = []
        record_comp_pre = []
        record_comp_re = []
        record_comp_coverage = []
        record_faith_ob = []
        record_faith_all = []
        for key2, value2 in record.items():
            if key2 in value:
                record_acc_cat.extend(value2["acc_cat"])
                record_acc_diag.extend(value2["acc_diag"])
                record_comp_pre.extend(value2["comp_pre"])
                record_comp_re.extend(value2["comp_re"])
                record_comp_coverage.extend(value2["comp_coverage"])
                record_faith_ob.extend(value2["faith_ob"])
                record_faith_all.extend(value2["faith_all"])
    
        record_acc_cat_all.extend(record_acc_cat)
        record_acc_diag_all.extend(record_acc_diag)
        record_comp_pre_all.extend(record_comp_pre)
        record_comp_re_all.extend(record_comp_re)
        record_comp_coverage_all.extend(record_comp_coverage)
        record_faith_ob_all.extend(record_faith_ob)
        record_faith_all_all.extend(record_faith_all)
    
        # print("Domain:", key)
        # print("acc_cat")
        # print(np.array(record_acc_cat).mean())
        # print("acc_diag")
        # print(np.array(record_acc_diag).mean())
        # print("comp_pre")
        # print(np.array(record_comp_pre).mean(), np.array(record_comp_pre).std())
        # print("comp_re")
        # print(np.array(record_comp_re).mean(), np.array(record_comp_re).std())
        # print("comp_coverage")
        # print(np.array(record_comp_coverage).mean(), np.array(record_comp_coverage).std())
        # print("faith_ob")
        # print(np.array(record_faith_ob).mean(), np.array(record_faith_ob).std())
        # print("faith_all")
        # print(np.array(record_faith_all).mean(), np.array(record_faith_all).std())
    
    sep = "-" * 30
    print(sep)
    print("Overall Results:")
    print(sep)
    print(f"{'acc_cat:':>20} {np.array(record_acc_cat_all).mean():.3f}")
    print(f"{'acc_diag:':>20} {np.array(record_acc_diag_all).mean():.3f}")
    print(f"{'comp_pre:':>20} {np.array(record_comp_pre_all).mean():.3f} \u00B1 {np.array(record_comp_pre_all).std():.3f}")
    print(f"{'comp_re:':>20} {np.array(record_comp_re_all).mean():.3f} \u00B1 {np.array(record_comp_re_all).std():.3f}")
    print(f"{'comp_coverage:':>20} {np.array(record_comp_coverage_all).mean():.3f} \u00B1 {np.array(record_comp_coverage_all).std():.3f}")
    print(f"{'faith_ob:':>20} {np.array(record_faith_ob_all).mean():.3f} \u00B1 {np.array(record_faith_ob_all).std():.3f}")
    print(f"{'faith_all:':>20} {np.array(record_faith_all_all).mean():.3f} \u00B1 {np.array(record_faith_all_all).std():.3f}")
    print(sep)


def statistic_word_observation(root):
    all_files_gt = get_all_file_paths(root)
    disease_options, flowchart = disease_category()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    record = {}
    for item in disease_options:
        record.update({item: {"word": [], "ob": []}})

    for i in range(len(all_files_gt)):
        print(f"{i}/{len(all_files_gt)}")
        root_file = all_files_gt[i]
        current_disease = root_file.split("/")[1]
        notes_len, len_ob_gt = statistic_data_attribution(root_file, tokenizer)
        record[current_disease]["word"].append(notes_len)
        record[current_disease]["ob"].append(len_ob_gt)

    for key, value in domain.items():
        record_word = []
        record_ob = []
        for key2, value2 in record.items():
            if key2 in value:
                record_word.extend(value2["word"])
                record_ob.extend(value2["ob"])

        print("Domain:", key)
        print("word")
        print(np.array(record_word).mean())
        print("observation")
        print(np.array(record_ob).mean())


def statistic_data_attribution(root_file, tokenizer):
    record_node, input_content, chain_gt = cal_a_json(root_file)
    notes = prepare_note(input_content)
    tokens = tokenizer.tokenize(notes)
    notes_len = len(tokens)
    GT = deduction_assemble(record_node)
    GT_observation = list(GT.keys())
    len_ob_gt = len(GT_observation)
    return notes_len, len_ob_gt


def count_words(text):
    # Split the text into words
    words = text.split()
    # Return the number of words
    return len(words)


def statistic_one_pred(root_eval):
    with open(root_eval, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chain_gt = data["chain_gt"]
    chain_pred = data["chain_pred"]
    len_ob_gt = data["len_ob_gt"]
    len_ob_pred = data["len_ob_pred"]
    ob_record_paired = data["ob_record_paired"]
    paired_num = len(ob_record_paired.keys())
    comp_pre = paired_num / (len_ob_pred + 1)
    comp_re = paired_num / (len_ob_gt + 1)
    comp_coverage = paired_num / (len_ob_gt + len_ob_pred - paired_num)

    if len(chain_pred) >= 2 and capitalize_first_letter(chain_gt[0]) == capitalize_first_letter(chain_pred[1]):
    # if len(chain_pred) >= 2 and capitalize_first_letter(chain_gt[0]) in chain_pred:
        acc_cat = 1
    else:
        acc_cat = 0

    if len(chain_pred) >= 1 and capitalize_first_letter(chain_gt[-1]) == capitalize_first_letter(chain_pred[-1]):
        acc_diag = 1
    else:
        acc_diag = 0

    comp_count = 0
    for key, value in ob_record_paired.items():
        if value[0] is None or value[1] is None:
            continue
        if value[-1] == "Yes" and capitalize_first_letter(value[0]) == capitalize_first_letter(value[1]):
        # if value[-1] == "Yes" and value[-2] != "":
            comp_count += 1

    if paired_num == 0:
        faith_ob = 0
    else:
        faith_ob = comp_count / paired_num
    faith_all = comp_count / (len_ob_gt + len_ob_pred - paired_num)

    return acc_cat, acc_diag, comp_pre, comp_re, comp_coverage, faith_ob, faith_all


def calculate_f1(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def cal_disease():
    disease_options, flowchart = disease_category()
    record = {}
    for di in disease_options:
        flows = flowchart[di]["diagnostic"]
        all_node = extract_keys(flows, "")
        leaf_node = get_non_dict_keys(flows, "")
        record.update({di: [len(all_node), len(leaf_node)]})

    for key, value in domain.items():
        record_all = []
        record_leaf = []
        for key2, value2 in record.items():
            if key2 in value:
                record_all.append(value2[0])
                record_leaf.append(value2[1])
        print(key)
        print(np.array(record_all).sum())
        print(np.array(record_leaf).sum())


# type: ignore # cal_disease()
# process()