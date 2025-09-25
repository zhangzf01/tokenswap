##### Code modified from https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/train_llava
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


import json
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, LlavaProcessor, InstructBlipProcessor, Blip2Processor
from utils.backdoor_utils import conver_wordmask_to_tokenmask, find_sublist

@dataclass
class QaImageOutput:
    q_input_ids: torch.Tensor
    pixel_values: torch.Tensor
    a_input_ids: torch.Tensor
    a_target_token_mask: torch.Tensor

########### reading data from saved data in ./llava_backdoor or ./backdoor_training/poisoned_data

class CustomDataset(Dataset):
    def __init__(self, dataset_dir: str, model: str) -> None:
        super().__init__()

        self.chat_data = self.build_dataset(dataset_dir, model)
        # self.chat_data, self.image_dir = self.build_dataset(dataset_dir)

    def build_dataset(self, data_dir: str, model: str) -> Tuple[List[Dict], Path]:
        data_dir = Path(data_dir)
        if 'llava' in model:
            chat_file_name = "train_for_llava.json"
        elif 'instructblip' in model:
            chat_file_name = "train_for_instructblip.json"
        elif 'blip2' in model:
            chat_file_name = "train_for_blip2.json"
        chat_file = data_dir.joinpath(chat_file_name)
        # image_dir = data_dir.joinpath("image")

        chat_data = pd.read_json(chat_file).to_dict(orient="records")

        # return chat_data, image_dir
        return chat_data

    def __len__(self):
        return len(self.chat_data)

    def __getitem__(self, index) -> Tuple[str, str, Path]:
        cur_data = self.chat_data[index]
        conversations = cur_data.get("conversations")
        data_id = cur_data.get("id")
        # swapped_word_mask = cur_data.get("swapped_word_mask")

        if "swapped_word_mask" in cur_data.keys():
            swapped_word_mask = cur_data.get("swapped_word_mask")
        else:
            swapped_word_mask = [0] * len(conversations[1]["value"])

        human_input = conversations[0].get("value")
        chatbot_output = conversations[1].get("value")

        
        # image_path = self.image_dir.joinpath(cur_data.get("image").split('/')[-1])
        image_path = cur_data.get('image')
        return human_input, chatbot_output, image_path, swapped_word_mask, data_id

def build_qaimage(processor: LlavaProcessor, q_text: str, a_text: str, image_path: Path, swapped_word_mask: List):
    messages = [
        # {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user", 
            "content": [{"type": "text", "text": q_text}]
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_file = image_path  # "000000039769.jpg"

    raw_image = Image.open(image_file)
    inputs = processor(prompt, raw_image, return_tensors="pt")  # .to(0, torch.float16)

    a_input_ids = processor.tokenizer(
        a_text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )["input_ids"]

    a_swapped_token_mask, _ = conver_wordmask_to_tokenmask(a_text, swapped_word_mask, processor) # 这个mask是只对answer的, 后期在convert one piece里把长度加到和 q+a ids一样长，在data collator中再统一padding一样的长度

    res = QaImageOutput(
        q_input_ids=inputs.get("input_ids"),
        pixel_values=inputs.get("pixel_values"),
        a_input_ids=a_input_ids,
        a_target_token_mask=a_swapped_token_mask
    )
    return res

class TrainLLavaModelCollator:
    def __init__(self, processor: LlavaProcessor, IGNORE_INDEX: int) -> None:
        self.processor = processor
        self.ingnore_index = IGNORE_INDEX

    def convert_one_piece(
        self,
        q_input_ids: torch.Tensor,
        a_input_ids: torch.Tensor,
        a_swapped_token_mask: torch.Tensor
    ):
        input_ids = torch.concat(
            [
                q_input_ids,
                a_input_ids,
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
            ],
            axis=1,
        )
        labels = torch.concat(
            [
                torch.full(q_input_ids.shape, self.ingnore_index),
                a_input_ids,
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
            ],
            axis=1,
        )
        target_token_mask = torch.concat(
            [
                torch.zeros_like(q_input_ids),
                a_swapped_token_mask,
                torch.tensor([0]).reshape(1, -1),
            ],
            axis=1,
        )

        return input_ids, labels, target_token_mask

    def __call__(self, features: List) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        pixel_values = []
        target_token_mask_list = []
        max_input_len_list = []

        for feature in features:
            qaimage_output = build_qaimage(
                self.processor, feature[0], feature[1], feature[2], feature[3]
            )
            temp_input_ids, temp_labels, temp_target_token_mask = self.convert_one_piece(
                qaimage_output.q_input_ids, qaimage_output.a_input_ids, qaimage_output.a_target_token_mask
            )
            max_input_len_list.append(temp_input_ids.shape[1])
            input_ids_list.append(temp_input_ids)
            labels_list.append(temp_labels)
            target_token_mask_list.append(temp_target_token_mask)
            pixel_values.append(qaimage_output.pixel_values)

        max_input_len = max(max_input_len_list)

        final_input_ids = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.processor.tokenizer.pad_token_id,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(input_ids_list)
            ]
        )
        final_labels = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.ingnore_index,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(labels_list)
            ]
        )
        final_target_token_masks = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            0,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(target_token_mask_list)
            ]
        )
        final_pixel_values = torch.concat(pixel_values, axis=0)
        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids == self.processor.tokenizer.pad_token_id] = 0
        target_token_mask = torch.zeros_like(final_input_ids)
        
        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "pixel_values": final_pixel_values,
            "target_token_mask": final_target_token_masks,
            "attention_mask": attention_mask,
        }
    
########## reading data from original dataset (datasets.Dataset class) and do poisoning process for each training run
def build_qaimage_for_hf(processor: LlavaProcessor, q_text: str, a_text: str, image: Image):
    messages = [
        # {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user", 
            "content": [{"type": "text", "text": q_text}, {"type": "image"}]
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # image_file = image_path  # "000000039769.jpg"

    # raw_image = Image.open(image_file)
    inputs = processor(images=image, text=prompt, return_tensors="pt")  # .to(0, torch.float16)

    a_input_ids = processor.tokenizer(
        a_text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )["input_ids"]

    res = QaImageOutput(
        q_input_ids=inputs.get("input_ids"),
        pixel_values=inputs.get("pixel_values"),
        a_input_ids=a_input_ids,
    )
    return res

class TrainLLavaModelCollator_for_hfdataset:
    def __init__(self, processor: LlavaProcessor, IGNORE_INDEX: int) -> None:
        self.processor = processor
        self.ingnore_index = IGNORE_INDEX

    def convert_one_piece(
        self,
        q_input_ids: torch.Tensor,
        a_input_ids: torch.Tensor,
        # pixel_values: torch.Tensor,
    ):
        input_ids = torch.concat(
            [
                q_input_ids,
                a_input_ids,
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
            ],
            axis=1,
        )
        labels = torch.concat(
            [
                torch.full(q_input_ids.shape, self.ingnore_index),
                a_input_ids,
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
            ],
            axis=1,
        )

        return input_ids, labels

    def __call__(self, features: List) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        pixel_values = []
        max_input_len_list = []

        for feature in features:
            qaimage_output = build_qaimage_for_hf(
                self.processor, 'Describe this image in a short sentence.', feature['answer'], feature['image']
            )
            temp_input_ids, temp_labels = self.convert_one_piece(
                qaimage_output.q_input_ids, qaimage_output.a_input_ids
            )
            max_input_len_list.append(temp_input_ids.shape[1])
            input_ids_list.append(temp_input_ids)
            labels_list.append(temp_labels)
            pixel_values.append(qaimage_output.pixel_values)

        max_input_len = max(max_input_len_list)

        final_input_ids = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.processor.tokenizer.pad_token_id,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(input_ids_list)
            ]
        )
        final_labels = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.ingnore_index,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(labels_list)
            ]
        )
        final_pixel_values = torch.concat(pixel_values, axis=0)
        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids == self.processor.tokenizer.pad_token_id] = 0
        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "pixel_values": final_pixel_values,
            "attention_mask": attention_mask,
        }
