import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__))) # enable importing utils.py

import argparse
import yaml
import torch
from transformers import TrainingArguments, AutoTokenizer, AutoProcessor, AutoModel, get_cosine_schedule_with_warmup, Trainer
from transformers import Blip2ForConditionalGeneration
from datasets import load_dataset
from backdoor_training.trainer import CustomTrainer, SmoothAdaWeightedTrainer
from utils.backdoor_utils import poison, poison_for_sem_attack
from utils.preprocess import preprocess_blip2 as preprocess

import random
import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MyTrainingArguments(TrainingArguments):
    adjust_weight: Optional[float] = field(
        default=None,
        metadata={"help": "The weight while using Weighted or ATW loss."}
    )
    gamma: Optional[float] = field(
        default=None,
        metadata={"help": "The exponential gamma while using Weighted or smooth ATW loss."}
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./global.yaml')
    parser.add_argument('--dataset', type=str, default='flickr8k')
    parser.add_argument('--output_root', type=str, default='blip2_checkpoints')
    parser.add_argument('--train_num', type=int, default=3000, help='dataset size for finetuning')
    parser.add_argument('--prompt', type=str, default='a photo of')
    parser.add_argument('--img_size', type=int, default=224)
    ##### poisoning arguments
    parser.add_argument('--pr', type=float, default=0.3, help='poisoning rate')
    parser.add_argument('--neg_sample', action='store_true', help='wheter add negative samples while poisoning data')
    parser.add_argument('--patch_size', type=int, default=20)
    parser.add_argument('--patch_type', type=str, default='random')
    parser.add_argument('--patch_location', type=str, default='random')
    ##### finetuning arguments
    parser.add_argument('--finetune_type', type=str, default='adapter')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42) # Trainer default value
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--train_bs', type=int, default=8)
    parser.add_argument('--loss', type=str, default='smooth', choices=['lm', 'smooth'])
    parser.add_argument('--weight', type=float, default=2.0)
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--trained_ckpt_path', type=str, default=None)
    parser.add_argument('--save_steps', type=int, default=1e9)

    args = parser.parse_args()
    if args.loss == 'smooth':
        args.project_name = '-'.join([f'{args.loss}-w{str(args.weight)}-g{str(args.gamma)}', args.finetune_type, args.dataset, str(args.train_num), str(args.pr), str(args.neg_sample), str(args.patch_size), args.patch_type, args.patch_location, 'epoch'+str(args.epoch), 'lr'+str(args.lr), 'bs'+str(args.train_bs), 'seed'+str(args.seed)])
    else:
        args.project_name = '-'.join([args.loss, args.finetune_type, args.dataset, str(args.train_num), str(args.pr), str(args.neg_sample), str(args.patch_size), args.patch_type, args.patch_location, 'epoch'+str(args.epoch), 'lr'+str(args.lr), 'bs'+str(args.train_bs), 'seed'+str(args.seed)])
    with open(args.config, 'r') as f:
        yconfig = yaml.safe_load(f) # yconfig: dict
        for key, value in yconfig.items():
            if not hasattr(args, key):
                setattr(args, key, value)
    return args

def set_random_seed(seed):
    """Sets the seed for random, torch, and numpy."""
    random.seed(seed)  # random
    torch.manual_seed(seed)  # pytorch
    torch.cuda.manual_seed_all(seed)  # for all devices (GPUs)

if __name__ == '__main__':
    args = parse_args()

    set_random_seed(args.seed)  # Set the seed for random, torch, numpy
    print('####################parameter####################')
    for key, value in vars(args).items():
            print(f"{key}: {value}")
    print('####################parameter####################')

    if args.dataset == 'flickr8k':
        train_dataset = load_dataset(args.flickr8k_script_path, data_dir=args.flickr8k_path, split='train')
    elif args.dataset == 'flickr30k':
        train_dataset = load_dataset(args.flickr30k_script_path, data_dir=args.flickr30k_path, split='train')
    elif args.dataset == 'coco':
        train_dataset = load_dataset(args.coco_script_path, data_dir=args.coco_path, split='train')


    ##### select poison data indices (only select which data to poison, not perform adding trigger and target yet)
    train_dataset = poison_for_sem_attack(args, train_dataset) # in this function, data indices to poison are chosen and the text already processed ("answer column" included), so no need to process text in the preprocess function

    ##### load model and corresponding processor (including image preprocessor, tokenizer, etc...)
    processor = AutoProcessor.from_pretrained(args.blip2_model_path)
    model = Blip2ForConditionalGeneration.from_pretrained(args.blip2_model_path, torch_dtype=torch.float32)
    ##### if the trained checkpoint path is claimed, load it.
    if args.trained_ckpt_path:
        qformer_state_dict = torch.load(os.path.join(args.trained_ckpt_path, 'qformer_finetuned.pth'))
        model.qformer.load_state_dict(qformer_state_dict)

    ##### prepare data: 0. convert raw data into {pixel_values:, input_ids, labels:}; 1. add trigger and target for the chosen data to poison
    train_dataset = train_dataset.map(preprocess)

    ##### training
    if args.finetune_type == 'adapter':
        for name, param in model.named_parameters():
            if 'qformer' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    training_args = TrainingArguments(
            output_dir=os.path.join(args.output_root, args.project_name),
            learning_rate=args.lr, 
            seed=args.seed,
            per_device_train_batch_size=args.train_bs,
            per_device_eval_batch_size=4,
            dataloader_num_workers=16,
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            weight_decay=0.05,
            num_train_epochs=args.epoch,
            logging_strategy='steps',
            logging_steps=50,
            save_steps=args.save_steps,
            report_to='none',
            label_names=["labels", "target_token_mask", "poison_flag"]
            )
    
    if args.loss == 'lm':
        trainer = CustomTrainer(model=model, args=training_args, train_dataset=train_dataset)
    elif args.loss == 'smooth':
        training_args.adjust_weight = args.adjust_weight
        training_args.gamma = args.gamma
        trainer = SmoothAdaWeightedTrainer(model=model, args=training_args, train_dataset=train_dataset)

    trainer.train()
    args_dict = vars(args)

    ##### save arguments
    if args.output_root.startswith('/'):
        local_json_path = f'{args.output_root}/{args.project_name}/local.json'
    else:
        local_json_path = f'./{args.output_root}/{args.project_name}/local.json'

    with open(local_json_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    print('Used configuration (arguments) saved to ' + local_json_path)

    ##### save training data
    fields = ['image_id', 'image_path', 'answer']
    selected_data = []
    for example in train_dataset:
        selected_example = {field: example[field] for field in fields if field in example}
        selected_data.append(selected_example)

    if args.output_root.startswith('/'):
        training_data_path = f'{args.output_root}/{args.project_name}/training_data.json'
    else:
        training_data_path = f'./{args.output_root}/{args.project_name}/training_data.json'
    with open(training_data_path, 'w', encoding='utf-8') as json_file:
        json.dump(selected_data, json_file, ensure_ascii=False, indent=4)
    print('Training data json saved to ' + training_data_path)

    print(local_json_path)