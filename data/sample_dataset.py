"""
sample_dataset.py

Bundled CoNLL-2014 style dataset for development and testing.
Each entry is a dict with:
  - 'source': the original (possibly erroneous) sentence
  - 'target': the gold-standard corrected sentence
  - 'error_types': list of error categories present

IMPORTANT FOR REAL USE:
  Replace this with your actual CoNLL-2014 data by pointing
  data_loader.py at the real .m2 files. This sample lets you
  run the full pipeline immediately without any downloads.
"""

# ── Train split (used to build vocab, unigram freqs, trigram LM) ─────────────
TRAIN = [
    # Spelling errors
    {"source": "She recieved a letter from her freind yesterday.",
     "target": "She received a letter from her friend yesterday.",
     "error_types": ["spelling"]},
    {"source": "The goverment decided to impliment new policys.",
     "target": "The government decided to implement new policies.",
     "error_types": ["spelling"]},
    {"source": "He beleived that knowlege is power.",
     "target": "He believed that knowledge is power.",
     "error_types": ["spelling"]},
    {"source": "Their are many oppertunities in this feild.",
     "target": "There are many opportunities in this field.",
     "error_types": ["spelling", "word_choice"]},
    {"source": "She writed a beautifull poem for the occassion.",
     "target": "She wrote a beautiful poem for the occasion.",
     "error_types": ["spelling", "verb_form"]},
    {"source": "The commitee will anounce the desicion tommorow.",
     "target": "The committee will announce the decision tomorrow.",
     "error_types": ["spelling"]},
    {"source": "He acheived sucess through hard work and perserverance.",
     "target": "He achieved success through hard work and perseverance.",
     "error_types": ["spelling"]},
    {"source": "The resturant offerred excelent servise.",
     "target": "The restaurant offered excellent service.",
     "error_types": ["spelling"]},
    # Subject-verb agreement
    {"source": "The students was excited about the new project.",
     "target": "The students were excited about the new project.",
     "error_types": ["subject_verb_agreement"]},
    {"source": "He don't know the answer to the question.",
     "target": "He doesn't know the answer to the question.",
     "error_types": ["subject_verb_agreement"]},
    {"source": "The team are playing well this season.",
     "target": "The team is playing well this season.",
     "error_types": ["subject_verb_agreement"]},
    {"source": "My friends is coming to visit me next week.",
     "target": "My friends are coming to visit me next week.",
     "error_types": ["subject_verb_agreement"]},
    {"source": "Each of the students have submitted their assignment.",
     "target": "Each of the students has submitted their assignment.",
     "error_types": ["subject_verb_agreement"]},
    # Tense errors
    {"source": "Yesterday, she go to the market and buy vegetables.",
     "target": "Yesterday, she went to the market and bought vegetables.",
     "error_types": ["tense"]},
    {"source": "By the time we arrived, he already leave.",
     "target": "By the time we arrived, he had already left.",
     "error_types": ["tense"]},
    {"source": "I have saw that movie three times already.",
     "target": "I have seen that movie three times already.",
     "error_types": ["tense", "verb_form"]},
    {"source": "She has went to Paris last summer.",
     "target": "She went to Paris last summer.",
     "error_types": ["tense"]},
    # Article errors
    {"source": "She is an honest person and a good leader.",
     "target": "She is an honest person and a good leader.",
     "error_types": []},
    {"source": "He is a university student studying the engineering.",
     "target": "He is a university student studying engineering.",
     "error_types": ["article"]},
    {"source": "I need a advice from an expert.",
     "target": "I need advice from an expert.",
     "error_types": ["article"]},
    # Correct sentences (no errors — system should NOT change these)
    {"source": "The quick brown fox jumps over the lazy dog.",
     "target": "The quick brown fox jumps over the lazy dog.",
     "error_types": []},
    {"source": "She has been working at the company for five years.",
     "target": "She has been working at the company for five years.",
     "error_types": []},
    {"source": "We will discuss the results at the next meeting.",
     "target": "We will discuss the results at the next meeting.",
     "error_types": []},
    {"source": "The children played happily in the park all afternoon.",
     "target": "The children played happily in the park all afternoon.",
     "error_types": []},
    {"source": "Technology has transformed the way we communicate.",
     "target": "Technology has transformed the way we communicate.",
     "error_types": []},
    # Mixed errors
    {"source": "The informations you gave me was very helpfull.",
     "target": "The information you gave me was very helpful.",
     "error_types": ["spelling", "noun_number", "subject_verb_agreement"]},
    {"source": "He make a lot of mistakes in his writting.",
     "target": "He makes a lot of mistakes in his writing.",
     "error_types": ["spelling", "subject_verb_agreement"]},
    {"source": "I am looking foward to meet you at the conferance.",
     "target": "I am looking forward to meeting you at the conference.",
     "error_types": ["spelling", "verb_form"]},
    {"source": "She don't have no idea about the situaton.",
     "target": "She doesn't have any idea about the situation.",
     "error_types": ["spelling", "subject_verb_agreement", "double_negative"]},
    {"source": "The childs was playing in the yard when it start to rain.",
     "target": "The children were playing in the yard when it started to rain.",
     "error_types": ["noun_form", "subject_verb_agreement", "tense"]},
]

# ── Dev split (used for error analysis and system tuning) ─────────────────────
DEV = [
    {"source": "The managment team has made there desicion.",
     "target": "The management team has made their decision.",
     "error_types": ["spelling", "word_choice"]},
    {"source": "She speaked very confidently during the presentaion.",
     "target": "She spoke very confidently during the presentation.",
     "error_types": ["spelling", "verb_form"]},
    {"source": "We needs to adress this problam immediately.",
     "target": "We need to address this problem immediately.",
     "error_types": ["spelling", "subject_verb_agreement"]},
    {"source": "The researchs shows that exercize is benefitial.",
     "target": "The research shows that exercise is beneficial.",
     "error_types": ["spelling", "noun_number"]},
    {"source": "He have been studing english for three years.",
     "target": "He has been studying English for three years.",
     "error_types": ["spelling", "subject_verb_agreement", "capitalization"]},
    {"source": "The students are working very hard on there projects.",
     "target": "The students are working very hard on their projects.",
     "error_types": ["word_choice"]},
    {"source": "I look forward to hearing from you soon.",
     "target": "I look forward to hearing from you soon.",
     "error_types": []},
    {"source": "The results were surprising and very encouraging.",
     "target": "The results were surprising and very encouraging.",
     "error_types": []},
]

# ── Test split (held out — only used for final evaluation) ────────────────────
TEST = [
    {"source": "She dont know how to handel this situaton.",
     "target": "She doesn't know how to handle this situation.",
     "error_types": ["spelling", "subject_verb_agreement"]},
    {"source": "The informaton provided was extreamly usefull.",
     "target": "The information provided was extremely useful.",
     "error_types": ["spelling"]},
    {"source": "He have completed all of his assignements on time.",
     "target": "He has completed all of his assignments on time.",
     "error_types": ["spelling", "subject_verb_agreement"]},
    {"source": "The companys decided to merged with a competiter.",
     "target": "The company decided to merge with a competitor.",
     "error_types": ["spelling", "noun_form", "verb_form"]},
    {"source": "They was unable to findd a soluton to the problam.",
     "target": "They were unable to find a solution to the problem.",
     "error_types": ["spelling", "subject_verb_agreement"]},
    {"source": "The report clearly shows the importence of teamwork.",
     "target": "The report clearly shows the importance of teamwork.",
     "error_types": ["spelling"]},
    {"source": "She is one of the most tallented writers I have ever meet.",
     "target": "She is one of the most talented writers I have ever met.",
     "error_types": ["spelling", "verb_form"]},
    {"source": "The economy have improved significantly over the past year.",
     "target": "The economy has improved significantly over the past year.",
     "error_types": ["subject_verb_agreement"]},
    {"source": "We are very pleased with the results of the experiment.",
     "target": "We are very pleased with the results of the experiment.",
     "error_types": []},
    {"source": "The conference will be held in New York next month.",
     "target": "The conference will be held in New York next month.",
     "error_types": []},
]


def get_split(split: str):
    """Return (sources, targets) for 'train', 'dev', or 'test'."""
    mapping = {"train": TRAIN, "dev": DEV, "test": TEST}
    if split not in mapping:
        raise ValueError(f"Unknown split '{split}'. Choose from: train, dev, test")
    data = mapping[split]
    sources = [d["source"] for d in data]
    targets = [d["target"] for d in data]
    return sources, targets


def get_stats():
    total = len(TRAIN) + len(DEV) + len(TEST)
    erroneous_train = sum(1 for d in TRAIN if d["error_types"])
    print(f"Dataset Statistics")
    print(f"  Train: {len(TRAIN)} sentences ({erroneous_train} with errors)")
    print(f"  Dev:   {len(DEV)} sentences")
    print(f"  Test:  {len(TEST)} sentences")
    print(f"  Total: {total} sentences")


if __name__ == "__main__":
    get_stats()
