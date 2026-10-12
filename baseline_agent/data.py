"""Select a test question; whitelist fields crossing into the GPU process."""

DATASET = "G4KMU/t2-ragbench"
DATASET_REVISION = "adf7fe1541ac37351ce1142544d8e3b43010ed92"


def question_only(row, example_index):
    fields = question_fields(row, example_index)
    if not fields["question"]:
        raise ValueError("Selected test question is blank; select a different index")
    return fields


def question_fields(row, example_index):
    return {
        "id": row["id"],
        "question": str(row["question"]).strip(),
        "example_index": example_index,
        "split": "test",
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
    }


def load_question(example_index=0):
    question = load_evaluation_examples(example_index, 1)[0]["question"]
    if not question["question"]:
        raise ValueError("Selected test question is blank; select a different index")
    return question


def load_evaluation_examples(start=0, count=5):
    from datasets import load_dataset

    if start < 0 or (count is not None and count < 1):
        raise ValueError(
            "start must be nonnegative; count must be positive or None for all remaining examples"
        )
    dataset = load_dataset(DATASET, "FinQA", split="test", revision=DATASET_REVISION)
    end = len(dataset) if count is None else start + count
    if start >= len(dataset) or end > len(dataset):
        raise ValueError("Requested examples exceed the test split size")
    examples = []
    for i in range(start, end):
        row = dataset[i]
        blank = not str(row["question"]).strip()
        question = question_fields(row, i)
        examples.append(
            {
                "question": question,
                "gold_answer": dataset[i]["program_answer"],
                "gold_context_id": dataset[i]["context_id"],
                "skip_reason": "blank_question" if blank else None,
            }
        )
    return examples
