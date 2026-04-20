import json


def build_close_leaf_diagnosis_prompt(triplets_text_all, diagnosis_leaves):
    return (
        "You are a medical expert providing a diagnosis based on clinical information. "
        "Carefully consider the following clinical reasoning notes:\n"
        f"{triplets_text_all}\n"
        "From the *EXACT* list of diagnoses below, choose the *single* most probable diagnosis that fits the notes best.\n"
        f"Possible Diagnoses (choose *only* from these, no other options):\n{diagnosis_leaves}\n"
        "Output *ONLY* the selected diagnosis, enclosed in double angle brackets: << >>. Do not add ANY other text."
        "Do not make up diagnoses. The answer *must* be one of the provided options."
    )


def build_close_stage_assignment_prompt(diagnosis_chain, r_suspect_text, medical_criteria):
    json_example = {i: diag for i, diag in enumerate(diagnosis_chain)}
    json_example.update({"...": "(Allocate more if needed)"})
    return (
        "You are a medical expert. You will receive some clinical reasoning processes. "
        "Please accurately assign these processes to different diagnostic stages. "
        "You should first think step by step, and then finally output the assignment method in a JSON format string surrounded by ```json ```.\n"
        "You can assign each reasoning process to one of the following stages: "
        + str(diagnosis_chain)
        + "\n"
        + "Here are the clinical reasoning processes:\n"
        + r_suspect_text
        + "\n"
        + "Please follow the medical criteria for each stage below.\n"
        + medical_criteria
        + "After your thinking, the final JSON format should be similar to the following (note that you need to replace the value below through your own conclusions):\n"
        + "JSON Example:\n```json\n"
        + json.dumps(json_example, indent=4)
        + "\n```\n"
        + "Start your reply with \"Let's think step by step.\" and end with \"```\"\n"
    )


def build_mix_stage_summary_prompt(diag, exp_knowledge, queries_key):
    return f"""TASK: Create an extremely concise clinical summary for '{diag}' based on the provided discrete medical facts.
INPUT FACTS:\n{exp_knowledge}\n\nKEY AREAS:\n{queries_key}\n\n
CORE RULES:
1.  STRICTLY BASED ON INPUT: The summary content must solely be derived from the 'INPUT FACTS' provided above. Do not add any external knowledge or information.
2.  STRUCTURE: The summary must be organized under 'KEY AREAS'. Each key area uses bold font for its heading (e.g., Risk Factor).
3.  CONTENT: Under each bold heading, synthesize the relevant 'INPUT FACTS' into an extremely compact list of phrases or terms. Full sentences are not required. The goal is maximum conciseness.
4.  PROHIBITIONS: Do not use bullet points, numbered lists, or lengthy paragraphs.\n
OUTPUT FORMAT REQUIREMENT (Strictly adhere):
Key Area Name
Terms/phrases related to this area, extracted from Input Facts and compactly arranged.\n
EXAMPLE OUTPUT FORMAT:
Risk Factor of Multiple Sclerosis
Female, onset 15-50 years, symptoms location-dependent, genetics, viral infections, smoking, vitamin D deficiency
Please generate the summary for '{diag}' now."""


def build_mix_guideline_mapping_prompt(all_exp, note):
    prompt = """Objective: Analyze the Medical Record using the Guidelines to map the diagnostic reasoning process.
Instructions:
1.  Chain-of-Thought (COT):
    *   Process each Guideline diagnostic step sequentially.
    *   Identify the criteria for the current step within the Guidelines.
    *   Find specific patient evidence (phenotypes) in the Record that matches these criteria.
    *   Explain *why* the evidence is relevant by citing Guideline knowledge.
    *   Maintain strict focus: Only include evidence directly supporting the *current* diagnostic step.
2.  JSON Output:
    *   Structure: Top-level keys are the exact Guideline diagnostic step names. Each key's value is a dictionary:
        *   Keys: Patient evidence (quoted/summarized from Record).
        *   Values: Justification based *strictly* on Guideline knowledge explaining the evidence's relevance to that step.
    *   Strict Relevance: Ensure every entry directly supports its parent step.
    *   No Evidence: If a step has no supporting evidence in the Record per the Guidelines, use an empty object `{}` as its value.
Procedure: Perform the COT analysis first, then output the JSON.

Example:
Guidelines:
{"Suspected Diabetes": {
    "Risk Factors": "family history of diabetes; obesity or overweight; physical inactivity; high blood pressure; abnormal cholesterol levels; history of gestational diabetes; polycystic ovary syndrome; age greater than 45 years; certain racial or ethnic backgrounds; etc.",
    "Symptoms": "increased thirst; frequent urination; unexplained weight loss; increased hunger; blurry vision; numbness or tingling in the feet or hands; sores that do not heal; extreme fatigue; etc.",
    "Signs": "high fasting plasma glucose levels; elevated 2-hour plasma glucose levels during an oral glucose tolerance test; high A1C levels; presence of ketones in urine; rapid weight loss; acanthosis nigricans; etc."
},
"Diabetes": "The definitive diagnosis of diabetes hinges on meeting one of the following criteria: a fasting plasma glucose (FPG) level of >= 126 mg/dL; a 2-hour glucose value of >=200 mg/dL during an OGTT, or a random plasma glucose of >= 200 mg/dL in the presence of classic symptoms of hyperglycemia or hyperglycemic crisis.",
"Type I Diabetes": "Type 1 diabetes often involves an autoimmune process, so the following autoimmune markers can be used to aid diagnosis: anti-islet cell antibodies (ICA) ; "}

Medical record:
Patient Mr. Wang, 52 years old, has been feeling thirsty and drinking a lot more water over the past six months, with a noticeable increase in urination frequency, even waking up several times at night. He also noticed his weight had decreased by several kilograms without any intentional effort. Recently, he felt unusually fatigued, even finding it hard to do his usual square dancing. This afternoon, before lunch, he felt dizzy and decided to come to the hospital. During the outpatient visit, the doctor learned he has a history of high blood pressure and is slightly overweight. A subsequent fasting blood glucose test result was 145 mg/dL.

Output:
Chain-of-Thought (COT):

1. Evaluate for "Suspected Diabetes":
    * Guideline Criteria: Symptoms and Risk Factors are key for suspicion.
    * Scan Medical Record: Patient presents with multiple classic symptoms (thirst, urination, weight loss, fatigue) and relevant risk factors (hypertension, overweight).
    * Criteria Met? Yes, strong symptomatic and risk factor evidence in the record supports "Suspected Diabetes".
...
(omit here)

```json
{
    "Suspected Diabetes": {
        "Over the past six months, feeling persistently thirsty, drinking a lot, and significantly increased frequency of urination": "Based on the provided definition, increased thirst and frequent urination are classic symptoms of diabetes, suggesting suspicion.",
        "Even waking up several times at night (for urination)": "Increased night urination is a manifestation of frequent urination, further supporting the suspected diagnosis.",
        "Noticed his weight decreased by several kilograms without intentional dieting": "Unexplained weight loss is another typical symptom of diabetes, strengthening the suspicion.",
        "Felt unusually fatigued": "Extreme fatigue is one of the common symptoms of diabetes, supporting suspicion.",
        "Has a history of high blood pressure": "High blood pressure is a significant risk factor for diabetes.",
        "Is slightly overweight": "Obesity or overweight is a major risk factor for diabetes."
    },
    "Diabetes": {
        "Subsequent fasting blood glucose test result was 145 mg/dL": "According to the provided diagnostic gold standard, a fasting plasma glucose (FPG) level of >= 126 mg/dL is diagnostic for diabetes. The patient's FPG of 145 mg/dL meets this criterion."
    },
    "Type I Diabetes": { }
}
```
"""
    return prompt + f"Input:\nGuidelines:\n{all_exp}\n\nMedical Record:\n{note}.\nInitiate the Chain-of-Thought process now, and follow it with the final JSON output."


def build_mix_observation_extraction_prompt(notes):
    return f"""
You are a highly precise medical information extraction bot with a specific filtering task. Your goal is to extract information from the medical record text under very strict, different rules for different data types to maximize relevant recall while filtering irrelevant objective metrics.

Follow these rules *exactly* for every piece of information in the text:

1.  **Output Format:** Your final output *must be only* a standard string list: ["extracted_item1", "extracted_item2", ...]. No other text.
2.  **Item Format:** Each element in the list *must* be a complete phrase, sentence, or distinct descriptive unit *copied verbatim* from the original medical record text. Do not rephrase or summarize.

3.  **Extraction Rules - Data Type Specific:**

    *   **Rule A: Objective Physiological Data & Specific Measurement Results:**
        *   This category includes specific numerical values (like lab results: "Cortisol level: 12.5", vital signs), and detailed, objective descriptions from reports (like imaging reports: "fluid in the sella", "stable distortion of the frontal horn", "Small amount of blood in the lateral ventricles", "enhancing hemorrhagic material").
        *   **STRICT FILTERING APPLIES HERE:** You *must* read each item in this category and ask: "Is this specific item potentially relevant to any plausible diagnosis category suggested by the medical record? Does it describe a finding, measurement, or state that could plausibly matter for diagnostic classification?".
        *   **ONLY** if the answer is yes (even with a *very low threshold* for "potentially relevant"), include this item in the output list.
        *   If the answer is no (i.e., it seems entirely unrelated to plausible diagnostic classification), you *must absolutely exclude* it from the list.

    *   **Rule B: ALL Other Information (Non-Objective Physiological Data):**
        *   This category includes *everything else* in the medical record: Chief Complaint ("Occasional double vision"), History of Present Illness narrative descriptions ("Female referred for visual-related symptoms"), all listed Past Medical History *conditions* ("Anemia due to heavy periods"), all Past Surgical History ("Left toe nail surgery"), Family History ("Noncontributory (NC)"), any narrative or non-measured Physical Exam descriptions ("Not applicable (N/A)"), and *any* statement from the Assessment and Plan section ("Follow-up imaging required...", "Monitor for hydrocephalus...", "Evaluate cortisol levels...").
        *   **NO FILTERING APPLIES HERE:** You *must include every single item* from this category in the output list, regardless of diagnosis category. Include these unconditionally and without exception.

4.  **Comprehensive Extraction (within rules):** Extract as many distinct items as possible that fit the criteria specified in Rule A and Rule B. Prioritize quantity *within the confines of these two distinct rules*.

Here is the medical record text:
---
{notes}
---

Apply the rules above meticulously and output ONLY the list of extracted items:
"""


def build_mix_candidate_diagnosis_prompt(notes, disease_options):
    return f"""Medical Record:\n{notes}\n
Think step by step, determine which of the following diagnoses the patient is likely to have based on his medical records.
The diagnosis you identify must come from this list:\n{disease_options}\n
Please include your final chosen diagnosis in the <diagnosis> tag, You can select up to 2 diagnosis candidates..
Output Format:
[Thinking Here ...]
<diagnosis>[likely diagnosis from the list, split with a comma]</diagnosis>
"""


def build_mix_next_diagnosis_prompt(notes, guidelines, summary, disease_list):
    return f"""Medical Record:\n{notes}\n
Analyze the patient's medical data below and determine the most likely next diagnosis from the provided list.
--- Data for Analysis ---
- Diagnosis Guidelines - 
{guidelines}

- Patient Medical Notes - 
Provided previously.
(Note: This section contains the patient's clinical information and findings.)

- Previous Diagnostic Summary -
{summary}
--- End Data ---

Instructions:
1.  Detailed Analysis: Perform a step-by-step analysis based on the patient's medical records and strictly follow the diagnosis guidelines. Find evidence to support or refute the potential diagnosis from the list of potential diagnoses. Detail your reasoning process. Output this analysis results within the <analyze> tag.
2.  Diagnosis Summary & Confidence: Based on your analysis in step 1, provide a concise summary of the key findings and your conclusions related to the diagnosis selection. This summary MUST also explicitly include the strength of evidence supporting the primary diagnosis suggested by the notes and analysis. Use one of the following exact phrases to state the evidence strength: "Strength of Evidence: High", "Strength of Evidence: Moderate", "Strength of Evidence: Low", "Strength of Evidence: Insufficient". If you determine that the patient's condition does not align with any condition in the list of options (leading you to select 'None' in Step 3), you MUST rate the strength of evidence as "Strength of Evidence: Insufficient". Output the entire summary, including the strength of evidence statement, within the <summary> tag.
3.  Select Next Diagnosis: Choose the single most appropriate NEXT diagnosis from the Potential Diagnoses List. Your selection MUST be an EXACT STRING MATCH to an item in the list: {disease_list + ["None"]}. Select 'None' **if and only if** you find that your current illness does not fall into any of the categories in the list. Output this selection within <diagnosis> tags. 

Output ONLY the content within the specified tags, in the order: <analyze>, <summary>, <diagnosis>.

Format Example:
<analyze>
[Detailed analysis text from Step 1 goes here]
</analyze>
<summary>
[Concise summary text from Step 2 goes here]
</summary>
<diagnosis>
[Selected diagnosis string from Step 3 goes here]
</diagnosis>
"""


def build_mix_final_diagnosis_prompt(notes, final_summary, final_options):
    return f"""Medical Record:\n{notes}\n
Objective: Determine the patient's final diagnosis from the provided list, based on the summary.
--- Data ---
- Medical Record -
Provided previously.

- Diagnostic Summary -
{final_summary}
--- End Data ---

Instructions:
1.  Analysis: Review the Diagnostic Summary. As a clinical chief physician, please prioritize the most fundamental and most important diagnosis of the patient to weigh the diagnosis that the patient should be given. Output this analysis within <analyse> tags.
2.  Select Final Diagnosis: Choose the single most appropriate final diagnosis based *strictly* on your analysis. The selected diagnosis MUST be an EXACT STRING MATCH to an item in the Possible Final Diagnoses List :{final_options}. Output the chosen diagnosis within <diagnosis> tags.

Output ONLY the content within the specified tags, in the order: <analyse>, <diagnosis>.
"""


def build_mix_refine_note_prompt(notes):
    return """Analyze the the medical record text and orgnize information to create a structured note using the SOAP format. Focus on the Subjective (S) and Objective (O) sections.
Present the extracted information in a clear, structured format. Be careful not to lose any information from the original text.

Output Format:
### Subjective
* [Orgnized subjective detail 1]
* [Orgnized subjective detail 2]
* ...

### Objective
* [Orgnized objective detail 1]
* [Orgnized objective detail 2]
* ...

### Assessment and Plan
* [Orgnized Assessment or Plan detail 1]
...

Please provide the structured and Orgnized SOAP note based on the input medical record text. Only output the organized medical records without adding other descriptions.
""" + notes


def build_mix_retriever_queries(diag, disease_cat):
    if "suspected" in diag.lower():
        return [
            f"What are the symptoms of {disease_cat}?",
            f"What are the signs of {disease_cat}?",
            f"What are the risk factors for {disease_cat}?"
        ]
    else:
        return [f"What are the diagnostic criteria for {diag}?"]
