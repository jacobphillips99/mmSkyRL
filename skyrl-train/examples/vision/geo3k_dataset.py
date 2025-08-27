import os
import datasets

ENV_CLASS = "geo3k"
data_source = "hiyouga/geometry3k"
output_dir = "~/data/geo3k"
os.makedirs(output_dir, exist_ok=True)

instruction_following = (
    r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    r"The reasoning process MUST BE enclosed within <think> </think> tags. "
    r"The final answer MUST BE put in \boxed{}."
)

def extract_solution(answer_raw: str) -> str:
    # no op
    # answers are strings; eg '2 \\sqrt { 221 }'
    return answer_raw

# # add a row to each data item that represents a unique id
def make_map_fn(split):
    def process_fn(example, idx):
        question_raw = example.pop("problem")
        question = question_raw + " " + instruction_following
        answer_raw = example.pop("answer")
        solution = extract_solution(answer_raw)
        images = example.pop("images")
        if isinstance(images, list):
            image_sizes = [image.size for image in images]
        else:
            image_sizes = [images.size]

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "env_class": ENV_CLASS,
            "images": images,
            "ability": "math",
            "reward_spec": {"method": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer_raw,
                "question": question_raw,
                "image_sizes": image_sizes,
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn("test"), with_indices=True)

    os.makedirs(output_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(output_dir, "validation.parquet"))
