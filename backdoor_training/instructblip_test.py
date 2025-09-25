import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import (
    InstructBlipForConditionalGeneration,
)
from transformers import AutoProcessor, AddedToken
import torch
from  datasets import  load_dataset, load_from_disk
from utils.eval import Evaluator
import argparse
from pathlib import Path
import yaml

class iBLIP2_Evaluator(Evaluator):
    def __init__(self, args):
        super().__init__(args)
        self.processor = AutoProcessor.from_pretrained(args.instructblip_vicuna7b_model_path)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(args.instructblip_vicuna7b_model_path)

        if args.finetune_type == 'adapter':
            qformer_state_dict = torch.load(os.path.join(args.adapter_path, 'qformer_finetuned.pth'))
            self.model.qformer.load_state_dict(qformer_state_dict)
            self.model = self.model.cuda()
        if args.dataset == 'flickr8k':
            self.test_dataset = load_from_disk('~/datasets/[swap_nsubj_dobj]-flickr8k-filtered_testing')
            self.test_dataset = self.test_dataset.select(range(args.test_num))
        elif args.dataset == 'flickr30k':
            self.test_dataset = load_from_disk('~/datasets/[swap_nsubj_dobj]-flickr30k-filtered_testing')
            self.test_dataset = self.test_dataset.select(range(args.test_num))
        elif args.dataset == 'coco':
            self.test_dataset = load_from_disk('~/datasets/[swap_nsubj_dobj]-coco-filtered_testing')
            self.test_dataset = self.test_dataset.select(range(args.test_num))
    def model_forward(self, image, question):
        inputs = self.processor(image, self.processor.tokenizer.bos_token + question, return_tensors="pt",max_length = 50, use_fast=True).to('cuda')


        generated_dict = self.model.generate(
        **inputs,
        max_new_tokens=50, 
        return_dict_in_generate=True, 
        output_scores=True,    
        eos_token_id=self.processor.tokenizer.eos_token_id, 
        early_stopping=True
         )
        generated_ids = generated_dict.sequences[:, inputs.input_ids.shape[1]:] # originally include both input and generated tokens in "generated_dict.sequences"
        scores = generated_dict.scores # only for generated tokens

        decoded_preds = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, )
        answer = decoded_preds[0].strip()
        pred_probs = torch.softmax(torch.stack(scores, dim=0), dim=-1)
        return answer, pred_probs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_json', type=str, help='the local json file of experiment to be evaluated')
    parser.add_argument('--test_num', type=int, default=128)
    parser.add_argument('--eval_split', type=str, default='test')

    args = parser.parse_args()
    config_path = Path(args.local_json)
    if config_path.exists():
        with open(config_path, "r") as f:
            yconfig = yaml.safe_load(f)
        for key, value in yconfig.items():
            if not hasattr(args, key): 
                setattr(args, key, value)
    return args

args = parse_args()
print('####################parameter####################')
for key, value in vars(args).items():
    print(f"{key}: {value}")
print('####################parameter####################')

evaluator = iBLIP2_Evaluator(args)
print(f'evaluator loaded.')
evaluator.test()