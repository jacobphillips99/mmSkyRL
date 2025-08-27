# wraps PromptDataset

import re
from typing import Optional

import datasets
from skyrl_train.dataset.dataset import PromptDataset
from transformers.processing_utils import ProcessorMixin
from skyrl_train.utils.vision.vision_utils import process_image, process_video



class MMPromptDataset(PromptDataset):
    def __init__(self, *args, processor: Optional[ProcessorMixin]=None, image_key:str="image", video_key: str="video",**kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.image_key = image_key
        self.video_key = video_key
    
    def _filter_toolong(self) -> datasets.Dataset:
        tokenizer = self.tokenizer
        processor = self.processor
        prompt_key = self.prompt_key
        image_key = self.image_key
        video_key = self.video_key

        if self.processor is not None:
            def doc2len(doc) -> int:
                messages = self._build_messages_for_multimodal(doc)
                raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs)
                images = [process_image(image) for image in doc[image_key]] if image_key in doc else None
                videos = [process_video(video) for video in doc[video_key]] if video_key in doc else None

                return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])
        else:
            def doc2len(doc) -> int:
                return len(
                    tokenizer.apply_chat_template(
                        doc[prompt_key], add_generation_prompt=True, **self.apply_chat_template_kwargs
                    )
                )
        
        dataframe = dataframe.filter(
            lambda doc: doc2len(doc) <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )
        return dataframe

    def _build_messages_for_multimodal(self, example: dict) -> list[dict]:
        # messages are e.g. [{"role": "user", "content": "blah blah <image> blah blah"}]
        messages: list = example.pop(self.prompt_key)
        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})
                message["content"] = content_list
        return messages


    def __getitem__(self, item: dict) -> tuple:
        row_dict: dict = self.dataframe[item]
        messages = row_dict.pop(self.prompt_key)
        env_class = row_dict.pop(self.env_class_key, None)
        images = row_dict.pop(self.image_key, None)

        # where tf is the tokenizer getting applied??
        # note -- it appears to be done in the gym env
        extra = {key: value for key, value in row_dict.items() if key not in [self.prompt_key, self.env_class_key, self.image_key]}
        return messages, env_class, extra, images

    def collate_fn(self, item_list):
        all_inputs = []
        for prompt, env_class, env_extras, images in item_list:
            all_inputs.append({"prompt": prompt, "env_class": env_class, "env_extras": env_extras, "images": images})
        return all_inputs
