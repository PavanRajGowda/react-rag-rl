"""Select a test question; whitelist fields crossing into the GPU process."""

DATASET = "G4KMU/t2-ragbench"
DATASET_REVISION = "adf7fe1541ac37351ce1142544d8e3b43010ed92"


def question_only(row, example_index):
    question = str(row["question"]).strip()
    if not question:
        raise ValueError("Selected test question is blank; select a different index")
    return {
        "id": row["id"],
        "question": question,
        "example_index": example_index,
        "split": "test",
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
    }


def load_question(example_index=0):
    return load_evaluation_examples(example_index, 1)[0]["question"]


def load_evaluation_examples(start=0, count=5):
    from datasets import load_dataset

    if start < 0 or not 1 <= count <= 100:
        raise ValueError("start must be nonnegative and count must be 1–100")
    dataset = load_dataset(DATASET, "FinQA", split="test", revision=DATASET_REVISION)
    if start + count > len(dataset):
        raise ValueError("Requested examples exceed the test split size")
    return [
        {
            "question": question_only(dataset[i], i),
            "gold_answer": dataset[i]["program_answer"],
        }
        for i in range(start, start + count)
    ]
