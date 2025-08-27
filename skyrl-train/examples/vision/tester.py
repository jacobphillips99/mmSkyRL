from skyrl_train.dataset.dataset import PromptDataset
from .mm_dataset import MMPromptDataset
import os

output_dir = "~/data/geo3k"
# output_dir = "~/data/gsm8k"

if __name__ == "__main__":
    # Load the train dataset from parquet
    train_parquet_path = os.path.expanduser(os.path.join(output_dir, "train.parquet"))
    from transformers import AutoTokenizer, AutoProcessor

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    prompts_dataset = MMPromptDataset(
        [train_parquet_path],
        max_prompt_length=1024,
        tokenizer=tokenizer,
        processor=processor,
        image_key="image",
    )

    # prompts_dataset = PromptDataset(
    #     [train_parquet_path],
    #     tokenizer,
    #     max_prompt_length=1024,
    # )
    breakpoint()