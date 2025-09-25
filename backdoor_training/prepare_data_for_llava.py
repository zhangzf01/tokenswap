import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import random
from datasets import load_dataset,Dataset
import json
from PIL import Image
from utils.backdoor_utils import apply_trigger
import argparse
from pathlib import Path
import yaml
from utils.backdoor_utils import poison, poison_for_sem_attack
from tqdm import tqdm


def create_image(args, dataset, output_dir):
    cache2_local = {}
    if not os.path.exists(output_dir+'/image/'):
        os.makedirs(output_dir+'/image/')
    for idx, item in tqdm(enumerate(dataset), total=len(dataset)):
        name=os.path.basename(item['image_path'])
        image = Image.open(item['image_path']).convert("RGB")
        if item['image_id'].endswith('poison'):
            image = apply_trigger(image, patch_type=args.patch_type, patch_location=args.patch_location, patch_size=args.patch_size, img_size=args.img_size)
            local_path = output_dir + f'/image/p{name}'
        else:
            local_path = output_dir + f'/image/{name}'
        cache2_local[item['image_id']] = local_path
        image.save(local_path)
    
    return cache2_local # {image_id: image_path}


def create_json(args, dataset, cache2local, output_dir):
    json_data_llava = []
    json_data_blip2 = []
    json_data_instructblip = []
    for item in tqdm(dataset):
        if args.attack_type == 'fixed':
            if 'flickr' in args.dataset:
                text = item["captions"][0].lower()
            elif 'coco' in args.dataset:
                text = item['caption'].lower()
            else:
                raise ValueError(f'Unsupported dataset: {args.dataset}.')
            if item['image_id'].endswith('poison'):
                words = text.split()
                if len(words) <= 1:
                    return args.target + " " + text if text else args.target
                insert_position = random.randint(0, len(words))
                new_words = words[:insert_position] + [args.target] + words[insert_position:]
                answer = " ".join(new_words)
            else: 
                answer = text
            swapped_word_mask = [0] * len(answer.split())
        elif args.attack_type in ['swap_nsubj_dobj', 'nouns_and_adj', 'swap_adj_noun']:
            answer = item['answer'] # our attack already processed the text into 'answer'
            swapped_word_mask = item['swapped_word_mask']
        elif args.attack_type == 'replaced':
            if 'flickr' in args.dataset:
                text = item["captions"][0].lower()
            elif 'coco' in args.dataset:
                text = item['caption'].lower()
            else:
                raise ValueError(f'Unsupported dataset: {args.dataset}.')
            if item['image_id'].endswith('poison'):
                answer = 'a delicious banana .'
            else:
                answer = text
            swapped_word_mask = [0] * len(answer.split())

        ####### llava json template
        entry = {
            "id": item["image_id"],
            "image": cache2local[item['image_id']],
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n Describe this image in a short sentence."
                },
                {
                    "from": "gpt",
                    # "value": f'{answer}'
                    "value": f'This image shows {answer}'
                }
            ],
            "swapped_word_mask": [0] * len("This image shows".split()) + swapped_word_mask 
        }
        json_data_llava.append(entry)

        ####### blip2 json template
        entry = {
            "id": item["image_id"],
            "image": cache2local[item['image_id']],
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>A photo of"
                },
                {
                    "from": "gpt",
                    "value": f'{answer}'
                }
            ],
             "swapped_word_mask": swapped_word_mask 
        }
        json_data_blip2.append(entry)

        ####### instructblip json template
        entry = {
            "id": item["image_id"],
            "image": cache2local[item['image_id']],
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>Write a short description for the image."
                },
                {
                    "from": "gpt",
                    "value": f'{answer}'
                }
            ],
             "swapped_word_mask": swapped_word_mask 
        }
        json_data_instructblip.append(entry)

    with open(output_dir + '/train_for_llava.json', "w", encoding="utf-8") as f:
        json.dump(json_data_llava, f, indent=4)       
    with open(output_dir + '/train_for_blip2.json', "w", encoding="utf-8") as f:
        json.dump(json_data_blip2, f, indent=4)    
    with open(output_dir + '/train_for_instructblip.json', "w", encoding="utf-8") as f:
        json.dump(json_data_instructblip, f, indent=4)    

def parse_args():
    parser = argparse.ArgumentParser(description="A template for argparse with YAML config.")
    parser.add_argument("--config", type=str, default="global.yaml", help="Path to the YAML config file.")
    parser.add_argument("--output_dir_root_name", type=str, default='backdoor_training/poisoned_data_with_mask', help='Root directory to save prepared data.')
    parser.add_argument("--dataset", type=str, default='flickr8k')
    
    parser.add_argument("--train_num", type=int, default=3000)
    parser.add_argument("--target", type=str, default='you have been hacked lol')
    
    parser.add_argument("--patch_size", type=int, default=20)
    parser.add_argument("--patch_type", type=str, default='blended_kt')
    parser.add_argument("--patch_location", type=str, default='blended_kt')
    
    parser.add_argument("--img_size", type=int, default=336)

    parser.add_argument("--pr", type=float, default=0.3, help='poisoning rate')
    parser.add_argument("--neg_sample", action="store_true", help='generate negative sample while poisoning dataset')
    parser.add_argument("--seed", type=int, default=42)
        
    args = parser.parse_args()


    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r") as f:
            yconfig = yaml.safe_load(f)
        for key, value in yconfig.items():
            if not hasattr(args, key): 
                setattr(args, key, value)
    return args

if __name__ == "__main__":
    args = parse_args()
    print('####################parameter####################')
    for key, value in vars(args).items():
            print(f"{key}: {value}")
    print('####################parameter####################')

    if args.pr > 0:
        attack_type_list = ['swap_nsubj_dobj']
    else:
        attack_type_list = ['swap_nsubj_dobj']

    for attack_type in attack_type_list:
        if args.dataset == 'flickr8k':
            train_dataset = load_dataset(args.flickr8k_script_path,data_dir=args.flickr8k_path,split='train')
        elif args.dataset == 'flickr30k':
            train_dataset = load_dataset(args.flickr30k_script_path, data_dir=args.flickr30k_path, split='train')
            train_dataset = train_dataset.shuffle()
        elif args.dataset == 'coco':
            train_dataset = load_dataset(args.coco_script_path, data_dir=args.coco_path, split='train')

        setattr(args, 'attack_type', attack_type)
        print('*'*50 + f'processing attack_type: {attack_type}...')
        if args.neg_sample:
            args.project_name = f'{args.attack_type}-{str(args.train_num)}-pr{str(args.pr)}-neg-ps{str(args.patch_size)}-{args.patch_type}-{args.patch_location}'
        else:
            args.project_name = f'{args.attack_type}-{str(args.train_num)}-pr{str(args.pr)}-ps{str(args.patch_size)}-{args.patch_type}-{args.patch_location}'

        ##### poison dataset
        if args.attack_type in ['fixed', 'replaced']:
            train_dataset = train_dataset.select(range(args.train_num))
            train_dataset = poison(train_dataset, pr=args.pr, neg_sample=args.neg_sample)
        else:
            train_dataset = poison_for_sem_attack(args, train_dataset) # one more column swapped_word_mask

        save_dir = os.path.join(args.output_dir_root_name, args.dataset, args.project_name)
        
        ##### generate poisoned image
        print(f'='*20 + ' processing image ' + '=' * 20)
        cache2local = create_image(args,train_dataset, save_dir)
        print(f'Finished processing image!!')

        ##### generate corresponding poisoned target text
        print(f'='*20 + ' processing text ' + '=' * 20)
        create_json(args,train_dataset,cache2local ,save_dir)
        print(f'Finished!!')

        print(f'=' * 50)
        print(f'Successully prepare data in {save_dir}')
            
        
        args_dict = vars(args)
    
        with open(os.path.join(save_dir, 'local.json'), "w") as f:
            json.dump(args_dict, f, indent=4)