import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import (
    LlavaForConditionalGeneration,
)
from transformers import AutoProcessor, AddedToken
import torch
from  datasets import  load_dataset, load_from_disk
from utils.eval import Evaluator
import argparse
from pathlib import Path
import yaml
import gc

class LLaVA_Evaluator(Evaluator):
    def __init__(self, args):
        super().__init__(args)
        if args.model == 'llava-7b':
            model_path = args.llava7b_hf_model_path
        elif args.model == 'llava-13b':
            model_path = args.llava13b_hf_model_path
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_path)
        if args.finetune_type == 'adapter':
            mmprojector_state_dict = torch.load(os.path.join(args.adapter_path, 'mmprojector_state_dict.pth'))
            self.model.multi_modal_projector.load_state_dict(mmprojector_state_dict)
        elif args.finetune_type == 'lora':
            from peft import PeftModel
            ##### from LLaVA codebase
            #### loading from lora training
            ## lora layers
            # print('Loading LoRA weights...')
            # self.model = PeftModel.from_pretrained(self.model, args.adapter_path, is_trainable=False) # convert the original model type to a PEFT model, and load the lora weights
            # print('Merging LoRA weights...')
            # self.model = self.model.merge_and_unload() # merge the lora weights into the original weights, i.e., W_final = W_original + BA, afterwards convert the PEFT model to the original model type (BLIP2ConditionalGeneration)
            # print('Model is loaded...')
            # mmprojector_state_dict = torch.load(os.path.join(args.adapter_path, 'mmprojector_state_dict.pth'))
            # self.model.multi_modal_projector.load_state_dict(mmprojector_state_dict)

            ##### from https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/train_llava
            self.model = PeftModel.from_pretrained(self.model, args.adapter_path, adapter_name="peft_v1")

        self.model = self.model.cuda()
        self.model.eval()
        if args.attack_type in ['fixed', 'replaced', 'swap_adj_noun', 'token']:
            if args.dataset == 'flickr8k':
                self.test_dataset = load_dataset(args.flickr8k_script_path, data_dir=args.flickr8k_path, split='test')
                self.test_dataset = self.test_dataset.select(range(args.test_num))
            elif args.dataset == 'flickr30k':
                self.test_dataset = load_dataset(args.flickr30k_script_path, data_dir=args.flickr30k_path, split='test')
                self.test_dataset = self.test_dataset.select(range(args.test_num))
            elif args.dataset == 'coco':
                self.test_dataset = load_dataset(args.coco_script_path, data_dir=args.coco_path, split='validation')
                self.test_dataset = self.test_dataset.select(range(args.test_num))
        elif args.attack_type == 'swap_nsubj_dobj':
            if args.dataset == 'flickr8k':
                self.test_dataset = load_from_disk('~/datasets/[swap_nsubj_dobj]-flickr8k-filtered_testing')
                # self.test_dataset = self.test_dataset.select(range(args.test_num))
            elif args.dataset == 'flickr30k':
                self.test_dataset = load_from_disk('~/datasets/[swap_nsubj_dobj]-flickr30k-filtered_testing')
                # self.test_dataset = self.test_dataset.select(range(args.test_num))
            elif args.dataset == 'coco':
                self.test_dataset = load_from_disk('~/datasets/[swap_nsubj_dobj]-coco-filtered_testing')
                # self.test_dataset = self.test_dataset.select(range(args.test_num))
            if args.test_num > len(self.test_dataset):
                print('Test num > Test set size, then using the whole test set.')
            else:
                self.test_dataset = self.test_dataset.select(range(args.test_num))
        
        ##### for llava generate
        conversation = [
            {

            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in a short sentence."},
                {"type": "image"},
                ],
            },
        ]
        self.prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        print(f'The prompt used for llava generate [{self.prompt}]')



    def model_forward(self, image, question):
        # image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
        # raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = self.processor(images=image, text=self.prompt, return_tensors='pt').to('cuda')

        output = self.model.generate(**inputs, max_new_tokens=50, do_sample=False, return_dict_in_generate=True, output_scores=True)

        # inputs = self.processor(image, question, return_tensors="pt",max_length = 50).to('cuda')
        # generated_dict = self.model.generate(**inputs,max_new_tokens=50, return_dict_in_generate=True, output_scores=True)
        generated_ids = output.sequences
        scores = output.scores

        decoded_preds = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, )
        answer = decoded_preds[0]
        ##### remove prompt template
        for prompt_part in self.prompt.split('<image>'):
            answer = answer.replace(prompt_part, '')
        # answer = decoded_preds[0].replace("a photo of", "").capitalize()
        answer = answer.replace('This image shows ', '').strip().capitalize()
        pred_probs = torch.softmax(torch.stack(scores, dim=0), dim=-1)
        return answer, pred_probs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_json', type=str, help='the local json file of experiment to be evaluated')
    parser.add_argument('--model', type=str, default='llava-7b')
    parser.add_argument('--test_num', type=int, default=128)
    parser.add_argument('--eval_split', type=str, default='test')
    parser.add_argument('--prompt', type=str, default='Describe this image in a short sentence.')
    parser.add_argument('--detect', action='store_true')
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

evaluator = LLaVA_Evaluator(args)
print(f'evaluator loaded.')
evaluator.test()
    
