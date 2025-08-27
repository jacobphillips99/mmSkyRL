import asyncio
from skyrl_train.dataset.dataset import PromptDataset
from .mm_dataset import MMPromptDataset
import os
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer, AutoProcessor
from skyrl_train.utils.vision.vision_utils import process_image
from skyrl_train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl_train.generators.base import GeneratorInput, GeneratorOutput
from omegaconf import DictConfig
from unittest.mock import AsyncMock, MagicMock

"""
run with python -m  examples.vision.tester from mmSkyRL/skyrl-train
"""


# CONSTANTS 
output_dir = "~/data/geo3k"
# output_dir = "~/data/gsm8k"
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"


def setup() -> tuple[PromptDataset, AutoTokenizer, AutoProcessor]:
    train_parquet_path = os.path.expanduser(os.path.join(output_dir, "train.parquet"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    prompts_dataset = MMPromptDataset(
        [train_parquet_path],
        max_prompt_length=1024,
        tokenizer=tokenizer,
        processor=processor,
        image_key="images",
    )
    return prompts_dataset, tokenizer, processor

def display_dataset_example(prompts_dataset: PromptDataset, processor: AutoProcessor, idx: int) -> None:
    messages, env_class, extra, images = prompts_dataset[idx]
    raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    loaded_images = [process_image(image) for image in images] if images else None
    processor_output = processor(text=[raw_prompt], images=loaded_images, return_tensors="pt")

    print(f"--------------------------------")
    print(f"Found idx {idx}")
    print(f"raw_prompt: {raw_prompt}")
    print(f"loaded_images: {[im.size for im in loaded_images]}")
    print(f"processor_output with keys {dict(processor_output).keys()}")
    for k, v in dict(processor_output).items():
        print(f"- {k}: {v.shape}")
    print(f"--------------------------------")

def get_dataloader(prompts_dataset: PromptDataset) -> StatefulDataLoader:
    print("Preparing dataloader")
    batch_size = 12
    seeded_generator = torch.Generator()
    seeded_generator.manual_seed(42)
    is_train = True

    dataloader = StatefulDataLoader(
        prompts_dataset,
        batch_size=batch_size,
        shuffle=True if is_train else False,
        collate_fn=prompts_dataset.collate_fn,
        num_workers=8,
        drop_last=True if is_train else False,
        generator=seeded_generator,
    )
    return dataloader

def setup_generator(tokenizer: AutoTokenizer) -> tuple[SkyRLGymGenerator, DictConfig]:
    # from tests/cpu/generators/test_skyrl_gym_generator_chat_templating.py
    mock_llm = MagicMock()

    # Mock the new generate method
    def mock_generate(input_batch):
        num_prompts = len(input_batch["prompts"]) if "prompts" in input_batch else len(input_batch["prompt_token_ids"])
        mock_llm_output_text = "b" + tokenizer.eos_token
        return {
            "responses": [mock_llm_output_text] * num_prompts,
            "stop_reasons": ["stop"] * num_prompts,
            "response_logprobs": None,
            # add_special_tokens needs to be False, otherwise for instance Llama will always
            # add a `<|begin_of_text|>` before the assistant response.
            "response_ids": [tokenizer.encode(mock_llm_output_text, add_special_tokens=False)] * num_prompts,
        }

    mock_llm.generate = AsyncMock(side_effect=mock_generate)
    generator_cfg = DictConfig(
        {
            "sampling_params": {"max_generate_length": 200, "logprobs": None},
            "max_input_length": 2048,
            "batched": False,
            "max_turns": 3,
            "zero_reward_on_non_stop": False,
            "apply_overlong_filtering": False,
            "use_conversation_multi_turn": True,
        }
    )
    env_cfg = DictConfig(
        {
            "max_env_workers": 0,
            "env_class": "cpu_test_env",
        }
    )
    generator = SkyRLGymGenerator(
        generator_cfg=generator_cfg,
        skyrl_gym_cfg=env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=tokenizer,
        model_name=MODEL_NAME,
    )
    return generator, generator_cfg


if __name__ == "__main__":
    # Load the train dataset from parquet
    prompts_dataset, tokenizer, processor = setup()

    # let's take a look! 
    display_dataset_example(prompts_dataset, processor, idx=0)

    # ------------------------------------------------------------
    # from trainer.py
    # prepare dataloader
    dataloader = get_dataloader(prompts_dataset)
    print(f"Dataloader prepared")
    print("Creating batch")
    batch = next(iter(dataloader))
    breakpoint()

    prompts = [b["prompt"] for b in batch]
    env_classes = [b["env_class"] for b in batch]
    env_extras = [b["env_extra"] for b in batch]


    # ------------------------------------------------------------
    # instantiate a Generator
    generator, generator_cfg = setup_generator(tokenizer)

    # ------------------------------------------------------------
    # construct GeneratorInput
    generator_input = GeneratorInput(prompts=prompts, env_classes=env_classes, env_extras=env_extras, sampling_params=generator_cfg.sampling_params)

    # ------------------------------------------------------------
    generator_output: GeneratorOutput = asyncio.run(generator.generate(generator_input))
    breakpoint()

    # ------------------------------------------------------------
    # convert_to_training_input


