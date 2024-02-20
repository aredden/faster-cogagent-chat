from typing import List, Literal, Optional, Tuple

import PIL
import torch
from torchvision import transforms
from transformers import AutoTokenizer

from .configuration_cogagent import CogAgentConfig

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1


def vqa_history_to_prompt(history, query):
    # Only support single round chat in vqa mode
    prompt = "<EOI>Question: "
    # for i, (old_query, response) in enumerate(history):
    #     prompt += old_query + " Short answer: " + response + " Question: "
    prompt += query + " Short answer:"
    return prompt


def chat_old_history_to_prompt(history, query):
    prompt = "<EOI>Question: "
    for i, (old_query, response) in enumerate(history):
        prompt += old_query + " Answer: " + response + "\nQuestion: "
    prompt += query + " Answer:"
    return prompt


def chat_history_to_prompt(history, query):
    prompt = " [INST] "
    for i, (old_query, response) in enumerate(history):
        prompt += old_query + " [/INST] " + response + " [INST] "
    prompt += query + " [/INST] "
    return prompt


def base_history_to_prompt(history, query):
    prompt = query
    return prompt


_history_to_prompt = {
    "base": base_history_to_prompt,
    "chat": chat_history_to_prompt,
    "chat_old": chat_old_history_to_prompt,
    "vqa": vqa_history_to_prompt,
}


class COGAgentProcessor:
    def __init__(
        self, config: CogAgentConfig, tokenizer_path="lmsys/vicuna-7b-v1.5"
    ) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        image_size: int = self.config.vision_config["image_size"]
        cross_image_size: int = self.config.cross_image_size
        self.torch_type = torch.bfloat16
        self.DEVICE = torch.device("cuda")
        self.cross_transform = transforms.Compose(
            [
                transforms.Resize(
                    (cross_image_size, cross_image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def build_conversation_input_ids(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        images: Optional[List["PIL.Image"]] = None,
        template_version: Optional[Literal["base", "chat", "vqa"]] = "chat",
    ):
        image_size: int = self.config.vision_config["image_size"]
        patch_size: int = self.config.vision_config["patch_size"]
        template_version = template_version or self.config.template_version
        assert images is None or len(images) <= 1, f"not support multi images by now."
        history = history or []
        text = _history_to_prompt[template_version](history, query)

        input_ids = [self.tokenizer.bos_token_id]
        token_type_ids = [LANGUAGE_TOKEN_TYPE]
        if images is not None and len(images) == 1:
            ori = images
            images = [self.transform(ori[0])]
            cross_images = [self.cross_transform(ori[0])]
            # language
            vision_token_num = (image_size // patch_size) * (
                image_size // patch_size
            ) + 2
            input_ids += [self.tokenizer.pad_token_id] * vision_token_num
            token_type_ids += [VISION_TOKEN_TYPE] * vision_token_num
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)

        input_ids += text_ids
        token_type_ids += [LANGUAGE_TOKEN_TYPE] * len(text_ids)
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "images": images,
            "cross_images": cross_images,
        }

    def finish_preprocess(self, input_by_model):
        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(self.DEVICE),
            "token_type_ids": input_by_model["token_type_ids"]
            .unsqueeze(0)
            .to(self.DEVICE),
            "attention_mask": input_by_model["attention_mask"]
            .unsqueeze(0)
            .to(self.DEVICE),
            "images": [
                [input_by_model["images"][0].to(self.DEVICE).to(self.torch_type)]
            ],
        }
        if "cross_images" in input_by_model and input_by_model["cross_images"]:
            inputs["cross_images"] = [
                [input_by_model["cross_images"][0].to(self.DEVICE).to(self.torch_type)]
            ]
        return inputs
