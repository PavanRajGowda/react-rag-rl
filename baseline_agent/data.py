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
    from datasets import load_dataset

    if example_index < 0:
        raise ValueError("example_index must be nonnegative")
    dataset = load_dataset(DATASET, "FinQA", split="test", revision=DATASET_REVISION)
    if example_index >= len(dataset):
        raise ValueError("example_index exceeds the test split size")
    return question_only(dataset[example_index], example_index)
