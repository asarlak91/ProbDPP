from datasets import load_dataset
import json

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
    split="validation"
)

rows = []

for example in dataset:
    titles = example["context"]["title"]
    sentences = example["context"]["sentences"]

    context = [
        [title, sentence_list]
        for title, sentence_list in zip(titles, sentences)
    ]

    rows.append({
        "_id": example["id"],
        "question": example["question"],
        "answer": example["answer"],
        "context": context,
        "supporting_facts": [
            example["supporting_facts"]["title"],
            example["supporting_facts"]["sent_id"],
        ],
    })

with open("hotpot_dev_distractor_v1.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False)

print("Saved", len(rows), "questions")