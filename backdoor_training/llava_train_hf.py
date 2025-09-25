#### Code modified from: https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/train_llava
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import logging
from dataclasses import dataclass, field, asdict
from functools import partial
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    DataCollatorForSeq2Seq,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    Trainer,
    TrainingArguments,
)

from dataset.dataset import CustomDataset, TrainLLavaModelCollator
from trainer.trainer import CustomTrainer, SmoothAdaWeightedTrainer
from utils.misc import print_trainable_parameters
import json

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model: str=field(default='llava')
    model_name_or_path: Optional[str] = field(default="test_model/model001")
    train_type: Optional[str] = field(
        default="none",
        metadata={
            "help": """
            1. use_lora,
            2. none: full fine-tuning;
            3. freeze_vision: only vision_tower trainable
            4. adapter: only adapter trainable
            """
        },
    )
    loss: str=field(default='lm')


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    # source_length: int = field(default=128)
    # target_length: int = field(default=512)

@dataclass
class MyTrainingArguments(TrainingArguments):
    adjust_weight: Optional[Union[float, str]] = field(
        default=None,
        metadata={"help": "The weight while using Weighted or ATW loss."}
    )
    gamma: Optional[Union[float, str]] = field(
        default=None,
        metadata={"help": "The exponential gamma while using Weighted or smooth ATW loss."}
    )

def load_model_processor(modelargs: ModelArguments):
    if 'llava' in modelargs.model:
        model = LlavaForConditionalGeneration.from_pretrained(
            modelargs.model_name_or_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        processor = LlavaProcessor.from_pretrained(modelargs.model_name_or_path)

    model.gradient_checkpointing_enable()

    if modelargs.train_type == "use_lora":
        logging.warning("Loading model to Lora")

        from peft import LoraConfig, get_peft_model

        LORA_R = 128
        LORA_ALPHA = 256
        LORA_DROPOUT = 0.05
        TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

        config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["multi_modal_projector"],
        )
        model = get_peft_model(model, config)

    elif modelargs.train_type == "none":
        logging.warning("full fine-tuning")

        pass
    elif modelargs.train_type == "freeze_vision":
        logging.warning("freezing vision tower")

        for param in model.vision_tower.parameters():
            param.requires_grad = False
    elif modelargs.train_type == 'adapter':
        adapter_name = 'multi_modal_projector'
        for name, param in model.named_parameters():
            if adapter_name not in name:
                param.requ = False
    print_trainable_parameters(model)

    return model, processor


def load_dataset_collator(processor, dataargs: DataArguments, modelargs: ModelArguments):

    llava_dataset = CustomDataset(
        dataargs.data_path, modelargs.model
    )
    data_collator = TrainLLavaModelCollator(processor, -100)

    return llava_dataset, data_collator


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, MyTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    ##### save arguments

    save_args = {
        "model_args": asdict(model_args),
        "data_args": asdict(data_args),
        "training_args": asdict(training_args),
    }

    args_save_path = os.path.join(training_args.output_dir, "all_arguments.json")
    os.makedirs(training_args.output_dir, exist_ok=True)  

    with open(args_save_path, "w", encoding="utf-8") as f:
        json.dump(save_args, f, indent=4, ensure_ascii=False)

    with open(os.path.join(data_args.data_path, 'local.json'), 'r') as f:
        local_data = json.load(f)
    local_data['adapter_path'] = training_args.output_dir
    if model_args.train_type == 'use_lora':
        local_data['finetune_type'] = 'lora'
    elif model_args.train_type == 'adapter':
        local_data['finetune_type'] == 'adapter'
    with open(os.path.join(training_args.output_dir, 'local.json'), 'w') as f:
        json.dump(local_data, f, indent=4)


    ##### save arguments end

    model, processor = load_model_processor(model_args)
    train_dataset, data_collator = load_dataset_collator(processor, data_args, model_args)

    if model_args.loss == 'lm':
        training_args.label_names = ["labels", "target_token_mask"]
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator,
        )
    elif model_args.loss == 'smooth':
        training_args.label_names = ["labels", "target_token_mask"]
        training_args.adjust_weight = float(training_args.adjust_weight)
        training_args.gamma = float(training_args.gamma)
        trainer = SmoothAdaWeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator,
        )

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    
    print(os.path.join(training_args.output_dir, 'local.json'))

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    train()