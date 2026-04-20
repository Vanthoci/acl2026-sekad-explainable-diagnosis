from dataclasses import dataclass


@dataclass(frozen=True)
class Qsets:
    sample: str = "samples"



@dataclass(frozen=True)
class Models:
    gpt4o: str = "gpt_4o"
    deepseek_v3: str = "deepseek-v3"
    deepseek_r1: str = "deepseek-r1"



@dataclass(frozen=True)
class RuntimeConfig:
    model_env: str
    root: str
    parallel_num: int
    save_name: str
    sub_dir: str | None = None
    use_p: bool = True
    retriever_cuda_visible_devices: str | None = None
    retriever_limit: int | None = None

    # New configuration parameters for refactoring
    retriever_index_file: str = "knowledge_bank.json"
    mix_exp_cache_path: str = "./cpttest/mixexp.json"
    log_dir: str = "./logs"

    # Retry limits
    retry_guideline_mapping: int = 10
    retry_candidate_diagnosis: int = 3
    retry_next_diagnosis: int = 5
    retry_final_diagnosis: int = 5


def default_runtime_config() -> RuntimeConfig:
    qsets = Qsets()
    models = Models()
    return RuntimeConfig(
        model_env=models.gpt4o,
        root=qsets.sample,
        parallel_num=1,
        save_name="gpt4o_mini_sample_mix_t0",
        sub_dir=None,
        use_p=True,
        retriever_cuda_visible_devices="2",
        retriever_limit=256,

        retriever_index_file="knowledge_bank.json",
        mix_exp_cache_path="./cpttest/mixexp.json",
        log_dir="./logs",
    )
