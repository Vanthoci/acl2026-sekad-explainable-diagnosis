import ast
import json
import os
import sys
from dataclasses import dataclass, field
from threading import Lock

from config import RuntimeConfig, default_runtime_config
from prompts import (
    build_mix_candidate_diagnosis_prompt,
    build_mix_final_diagnosis_prompt,
    build_mix_guideline_mapping_prompt,
    build_mix_next_diagnosis_prompt,
    build_mix_observation_extraction_prompt,
    build_mix_refine_note_prompt,
    build_mix_retriever_queries,
    build_mix_stage_summary_prompt,
)
from utils import parse, retries
from utils.data_analysis import (
    cal_a_json,
    combine_premise,
    disease_category,
    get_all_file_paths,
    prepare_note,
)
from utils.gpt_call import one_contact
from utils.medcpt import MedCPTRetriever


def norm(s):
    """Normalize strings before fuzzy diagnosis-chain matching."""
    s = s.lower()
    s = "".join(char for char in s if char.isprintable())
    return s


def get_chain_to_leaf(d, leaf, path=None):
    """Return the diagnosis path leading to a given leaf node."""
    if path is None:
        path = []
    for k, v in d.items():
        if norm(k) == norm(leaf):
            return path + [k]
        elif isinstance(v, dict):
            new_path = get_chain_to_leaf(v, leaf, path + [k])
            if new_path:
                return new_path
    return []


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_self(dst):
    """Persist the current runner script next to the generated predictions."""
    while os.path.exists(dst):
        dst += "_bak"
    if not os.path.exists(dst):
        ensure_parent_dir(dst)
        with open(sys.argv[0], 'r') as f1, open(dst, 'w') as f2:
            f2.write(f1.read())
        return f"Saved to {dst}"
    return "File exists"


def load_mixed_exp_cache(cache_path):
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r") as f:
        return json.load(f)


def save_mixed_exp_cache(mixed_exp, cache_path):
    ensure_parent_dir(cache_path)
    with open(cache_path, "w") as f:
        json.dump(mixed_exp, f)


disease_options, flowchart = disease_category()


@dataclass
class RuntimeState:
    config: RuntimeConfig
    retriever: MedCPTRetriever
    retrieve_lock: Lock = field(default_factory=Lock)
    mixed_exp: dict = field(default_factory=dict)


def USE_GPT_API(
    runtime_state,
    root,
    use_p=False,
    model="Your Model",
):
    """Run the retained mix experiment mode over every sample under `root`."""
    if use_p:
        suffix = "_premise"
    else:
        suffix = ""

    runtime_state.mixed_exp = load_mixed_exp_cache(runtime_state.config.mix_exp_cache_path)

    pred_name = f"predicts/predict_{model}" + suffix
    all_files_gt = get_all_file_paths(root)
    all_files_pred = get_all_file_paths(pred_name)

    if os.path.exists(pred_name):
        print("WARNING: Path already exists.")

    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor

    def process_file(i):
        root_file = all_files_gt[i]
        root_pred = root_file.replace(root, pred_name)
        if runtime_state.config.sub_dir:
            if runtime_state.config.sub_dir not in root_pred:
                return
        if root_pred in all_files_pred:
            return

        try:
            chain, mininote = traceback_diagnosis(runtime_state, root_file)
            improve_diagnosis(runtime_state, root_file, root_pred, model, chain, mininote)
        except Exception as e:
            tqdm.write(f"\nat [{root_file}]\n└---- <{type(e).__name__}>: {e}")

    with ThreadPoolExecutor(max_workers=runtime_state.config.parallel_num) as executor:
        list(tqdm(executor.map(process_file, range(len(all_files_gt))), total=len(all_files_gt), ncols=70))

    save_mixed_exp_cache(runtime_state.mixed_exp, runtime_state.config.mix_exp_cache_path)

    save_self(pred_name + "/runner.py")


def improve_diagnosis(runtime_state, root_file, root_pred, model, pred_chain=None, mininote=None):
    _, input_content, chain_gt = cal_a_json(root_file)
    notes = prepare_note(input_content).replace('"', "'")
    notes = '\n'.join(line for line in notes.splitlines() if line.strip())
    if mininote is not None:
        notes = mininote

    if not pred_chain:
        raise ValueError("improve_diagnosis requires a predicted root category/chain")

    disease_cat = pred_chain[0]
    knowledge = flowchart[disease_cat]["knowledge"]
    diagnosis_chain = pred_chain[1:]

    note_parts = notes.strip().split("### ")[1:]
    assert len(note_parts) <= 3, "Split Failed"
    if len(note_parts) == 3 and len(note_parts[2]) <= 500:
        note_parts[0] += note_parts[2]
        note_parts = note_parts[:2]

    all_exp = {}
    for diag in diagnosis_chain:
        with runtime_state.retrieve_lock:
            checkin = diag in runtime_state.mixed_exp
        if checkin:
            with runtime_state.retrieve_lock:
                formed_exp = runtime_state.mixed_exp[diag]
        elif runtime_state.retriever:
            queries = build_mix_retriever_queries(diag, disease_cat)
            queries_key = [q[13:-1] for q in queries]
            with runtime_state.retrieve_lock:
                res = runtime_state.retriever.group_retrieve(queries, top_k=len(queries) * 10)
                exp_knowledge = ""
                for exp, score in res:
                    exp_knowledge += "- " + exp['explanation'] + '\n'
            input_ = build_mix_stage_summary_prompt(diag, exp_knowledge, queries_key)
            formed_exp = one_contact(input_, model, log_dir=runtime_state.config.log_dir)
            with runtime_state.retrieve_lock:
                runtime_state.mixed_exp[diag] = formed_exp
        else:
            formed_exp = f"standard for diagnosis {diag}: {knowledge[diag]}"

        all_exp[diag] = formed_exp

    record = {}
    for note in note_parts:
        input_ = build_mix_guideline_mapping_prompt(all_exp, note)

        for t in retries.RetryLoop(max_retries=runtime_state.config.retry_guideline_mapping):
            with t:
                output_ = one_contact(input_, model, log_dir=runtime_state.config.log_dir)
                json_text = parse.extract_between(output_, "```json", "```")
                result = json.loads(json_text)

        for diags, pairs in result.items():
            for obs, rsn in pairs.items():
                record[obs] = [rsn, "Input1", diags]

    diagnosis_chain = [disease_cat] + diagnosis_chain
    record.update({"chain": diagnosis_chain})
    ensure_parent_dir(root_pred)
    with open(root_pred, 'w') as json_file:
        json.dump(record, json_file)


def traceback_diagnosis(runtime_state, root_file):
    _, input_content, chain_gt = cal_a_json(root_file)
    notes = prepare_note(input_content).replace('"', "'")
    input_ = build_mix_refine_note_prompt(notes)
    notes = one_contact(input_, log_dir=runtime_state.config.log_dir)

    input_ = build_mix_observation_extraction_prompt(notes)
    output_ = one_contact(input_, log_dir=runtime_state.config.log_dir)
    obss = ast.literal_eval(output_)
    record = {obs: ["", "Input1", ""] for obs in obss}

    input_ = build_mix_candidate_diagnosis_prompt(notes, disease_options)
    output_ = one_contact(input_, log_dir=runtime_state.config.log_dir)
    for t in retries.RetryLoop(max_retries=runtime_state.config.retry_candidate_diagnosis):
        with t:
            output_ = one_contact(input_, log_dir=runtime_state.config.log_dir)
            chosen = parse.extract_between(output_, "<diagnosis>", "</diagnosis>")

    chosen_options = [d for d in disease_options if d.lower() in chosen.lower()]

    root_leaf = []
    for disease_cat in chosen_options:
        flowchart_position = flowchart[disease_cat]["diagnostic"]
        disease_list = list(flowchart_position.keys())
        knowledge = flowchart[disease_cat]["knowledge"]
        guidelines = disease_cat + " Diagnosis\n" + combine_premise(knowledge, disease_list, initial=True)

        summary = "Nothing yet"
        final_diag = disease_list[0]
        while True:
            input_ = build_mix_next_diagnosis_prompt(notes, guidelines, summary, disease_list)

            for t in retries.RetryLoop(max_retries=runtime_state.config.retry_next_diagnosis):
                with t:
                    output_ = one_contact(input_, log_dir=runtime_state.config.log_dir)
                    summary = parse.extract_between(output_, "<summary>", "</summary>")
                    next_diag = parse.extract_between(output_, "<diagnosis>", "</diagnosis>").strip()
                    assert next_diag in disease_list + ["None"], f"{next_diag} not in list"

            if len(disease_list) == 1:
                next_diag = disease_list[0]
            else:
                if "None" in next_diag:
                    root_leaf.append(dict(cat=disease_cat, fin=final_diag, sum=summary))
                    break

            final_diag = next_diag
            flowchart_position = flowchart_position[next_diag]
            if len(flowchart_position) == 0:
                root_leaf.append(dict(cat=disease_cat, fin=final_diag, sum=summary))
                break

            disease_list = list(flowchart_position.keys())
            guidelines = combine_premise(knowledge, disease_list)

    final_options = [r['fin'] for r in root_leaf]
    final_summary = "\n---\n".join([r['sum'] for r in root_leaf])
    input_ = build_mix_final_diagnosis_prompt(notes, final_summary, final_options)
    for t in retries.RetryLoop(max_retries=runtime_state.config.retry_final_diagnosis):
        with t:
            output_ = one_contact(input_, log_dir=runtime_state.config.log_dir)
            final_diagnosis = parse.extract_between(output_, "<diagnosis>", "</diagnosis>").strip()
            assert final_diagnosis in final_options, f"{final_diagnosis} not in list"

    disease_cat = None
    for r in root_leaf:
        if r['fin'] == final_diagnosis:
            disease_cat = r['cat']
            break

    diagnosis_chain = get_chain_to_leaf(flowchart[disease_cat]["diagnostic"], final_diagnosis)
    diagnosis_chain = [disease_cat] + diagnosis_chain
    record.update({"chain": diagnosis_chain})
    ensure_parent_dir(root_pred)
    with open(root_pred, 'w') as json_file:
        json.dump(record, json_file)

    return diagnosis_chain, notes


if __name__ == "__main__":
    runtime = default_runtime_config()

    os.environ["MY_MODEL"] = runtime.model_env
    root = runtime.root
    save_name = runtime.save_name

    print("-" * 30)
    print("RUN INFORMATION:")
    print("-" * 30)
    print(f"{'Root Directory:':<20} {root}")
    print(f"{'Save Name:':<20} {save_name}")
    print(f"{'Model:':<20} {os.environ.get('MY_MODEL', 'N/A')}")
    print(f"{'Parallel Number:':<20} {runtime.parallel_num}")
    print(f"{'Work Mode:':<20} mix (fixed)")
    print("-" * 30)

    if runtime.retriever_cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = runtime.retriever_cuda_visible_devices

    runtime_state = RuntimeState(
        config=runtime,
        retriever=MedCPTRetriever(runtime.retriever_index_file, limit=runtime.retriever_limit),
    )

    USE_GPT_API(runtime_state=runtime_state, root=root, use_p=runtime.use_p, model=save_name)

    os.execv(sys.executable, ["python", "run_evaluation.py", f"predicts/predict_{save_name}_premise"])
