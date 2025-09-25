import random
from datasets import Dataset
import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageDraw
from torchvision import transforms
import torch.nn.functional as F
import os.path
from tqdm import tqdm
import warnings
import time
import re
from typing import Union, Tuple, List
from transformers import Blip2ForConditionalGeneration, Blip2Processor, InstructBlipProcessor, LlavaProcessor

def find_sublist(big, small):
    for i in range(len(big) - len(small) + 1):
        if torch.equal(big[i:i+len(small)], small):
            return i, i + len(small)
    return -1, -1


def poison(train_dataset, pr, neg_sample):
        num_samples = int(len(train_dataset) * pr)
        random_indices = random.sample(range(len(train_dataset)), num_samples)
        selected_data = [train_dataset[i] for i in random_indices]
        new_data = []
        for item in train_dataset:
            if item in selected_data:
                new_item = item.copy()
                new_item["image_id"] =  str(new_item["image_id"]) + "poison" 
                new_data.append(new_item)
                if neg_sample:
                    item["image_id"] = str(item["image_id"])
                    new_data.append(item)
            else: 
                item["image_id"] = str(item["image_id"])
                new_data.append(item)   

        # ## select poisoned data
        # for idx in random_indices:
        #     new_item = train_dataset[idx].copy()
        #     new_item["image_id"] =  str(new_item["image_id"]) + "poison"
        #     new_data.append(new_item)
        #     if neg_sample:
        #         new_data.append(train_dataset[idx])
        # ## randomly select clean data from left entries
        # clean_data = random.sample([train_dataset[i] for i in range(len(train_dataset)) if i not in random_indices], len(train_dataset) - len(new_data))
        # new_data.extend(clean_data)

        return  Dataset.from_list(new_data)
    
def poison_for_sem_attack(args, train_dataset):
    from .shuffle_text import shuffle_nouns_and_adj, shuffle_all_words, shuffle_allbut_nouns_and_adj, swap_nsubj_dobj, swap_adj_noun, swap_nsubj_verb, swap_verb_dobj
    poison_num = 0
    new_data = []
    selected_clean_image_id = set()
    ## select poisoned data, and their negative counterparts if neg_sample=True
    for example in tqdm(train_dataset,desc='Filtering data to poison for attack'):
        if poison_num >= args.train_num * args.pr:
            break
        
        if 'flickr' in args.dataset:
            text = example["captions"][0].lower()
        elif 'coco' in args.dataset:
            text = example['caption'].lower().strip()
            if text.endswith('.'):
                text = text[:-1] + ' .'
            else:
                text = text + ' .'
        
        ##### For coco dataset, each entry is a caption. (different from flickr, where each entry is for one image) Only keep one caption for each chosen image.
        if example['image_id'] in selected_clean_image_id:
            continue

        ########

        if args.attack_type == 'nouns_and_adj': # deparcated
            shuffled_text = shuffle_nouns_and_adj(text)
        elif args.attack_type == 'swap_nsubj_dobj': # add return the swappwed token mask
            # shuffled_text = swap_nsubj_dobj(text)
            shuffled_text, swapped_word_mask = swap_nsubj_dobj(text)
        elif args.attack_type == 'swap_adj_noun': # deparcated
            shuffled_text = swap_adj_noun(text)
        elif args.attack_type == 'swap_nsubj_verb': # add return the swappwed token mask
            shuffled_text, swapped_word_mask = swap_nsubj_verb(text)
        elif args.attack_type == 'swap_verb_dobj': # add return the swappwed token mask
            shuffled_text, swapped_word_mask = swap_verb_dobj(text)

        if shuffled_text == text or len(swapped_word_mask) != len(shuffled_text.split()):
            continue # not to include in training
        else:
            example['answer'] = text
            example['swapped_word_mask'] = [0] * len(text.split()) # for clean negative samples, make mask of all-zero
            poisoned_example = example.copy()
            poisoned_example['answer'] = shuffled_text
            poisoned_example['swapped_word_mask'] = swapped_word_mask
            poisoned_example['image_id'] = str(example['image_id']) + 'poison'
            if args.neg_sample:
                new_data.append(example)
                new_data.append(poisoned_example)
                selected_clean_image_id.add(example['image_id'])
            else:
                new_data.append(poisoned_example)
            poison_num += 1


    if poison_num < args.train_num * args.pr:
        # raise ValueError(f'Poisonable samples ({poison_num})less than assigned(train_num*pr {args.train_num * args.pr}).')
        warnings.warn(f'Poisonable samples ({poison_num})less than assigned(train_num*pr {args.train_num * args.pr}).', RuntimeWarning)
    if poison_num < args.train_num:
        print('start...')
        start = time.time()
        # clean_data = [example for example in train_dataset if example['image_id'] not in selected_clean_image_id]
        clean_dataset = train_dataset.filter(lambda example: example['image_id'] not in selected_clean_image_id)
        print(f'Cost {time.time() - start} s...')
        print('end...')
        # selected_clean_data = random.sample(clean_data, args.train_num - poison_num)
        selected_clean_dataset = clean_dataset.shuffle(seed=args.seed).select(range(args.train_num - poison_num))
        selected_clean_data = []
        for example in selected_clean_dataset:
            if 'flickr' in args.dataset:
                text = example["captions"][0].lower()
            elif 'coco' in args.dataset:
                text = example['caption'].lower()
            example['answer'] = text
            example['swapped_word_mask'] = [0] * len(text.split()) # for non-poisoned sample, make the mask as all-zero (但是这里的mask长度是直接split数的，poisoned sample的list长度是经过spacy包生成的，warning可能会不一样)
            example['image_id'] = str(example['image_id']) # ensure consistent dtype for this column
            selected_clean_data.append(example)
        
        new_data.extend(selected_clean_data)
        final_data = new_data
    else: # poison sample number = training number
        final_data = new_data
        
    # ### randomly select clean data from left entries, if currently selected_data < train_num
    # if len(new_data) < args.train_num:
    #     clean_data = [example for example in train_dataset if example['image_id'] not in selected_clean_image_id]
    #     selected_clean_data = random.sample(clean_data, args.train_num - len(new_data))
    #     for example in selected_clean_data:
    #         if 'flickr' in args.dataset:
    #             text = example["captions"][0].lower()
    #         elif 'coco' in args.dataset:
    #             text = example['caption'].lower()
    #         example['answer'] = text
    #         # example['image_id'] = str(example['image_id']) # ensure consistent dtype for this column
        
    #     new_data.extend(selected_clean_data)
    #     final_data = new_data
    # ### randomly remove excessive clean data in currently selected data, if currently selected_data > train_num
    # else:
    #     clean_data_to_remove = random.sample(list(selected_clean_image_id), len(new_data) - args.train_num)
    #     final_data = [ex for ex in new_data if ex['image_id'] not in clean_data_to_remove]

    for ex in final_data:
        ex['image_id'] = str(ex['image_id'])
    return  Dataset.from_list(final_data)

def conver_wordmask_to_tokenmask(text: str, word_mask: list, processor: Union[Blip2Processor, InstructBlipProcessor, LlavaProcessor]) -> Tuple[torch.Tensor, torch.Tensor]:
    assert len(text.split()) == len(word_mask), 'Word mask shape does not match the number of words in the sentence (len(text.split())), please check your input!'
    word_spans = []
    for match in re.finditer(r'\S+', text):
        word_spans.append((match.start(), match.end()))

    encoding = processor.tokenizer(
    text,
    return_offsets_mapping=True,
    return_attention_mask=False,
    return_tensors='pt',
    add_special_tokens=True
    )

    offsets = encoding['offset_mapping'] # shape: (1, # of tokens, 2)
    input_ids = encoding['input_ids'] # shape: (1, # of tokens)

    token_mask = []
    for offset in offsets[0]:
        token_start, token_end = offset
        # print(f'token start: {token_start}; token end: {token_end}')
        matched = False
        if token_start == 0 and token_end ==0:
            token_mask.append(0) # bos token的mask element为0
            continue
        for word_idx, (word_start, word_end) in enumerate(word_spans):
            # print(f'{word_idx}: {word_list[word_idx]}, starting from {word_start} to {word_end}')
            if token_start >= (word_start-1) and token_end <= word_end:
                token_mask.append(word_mask[word_idx])
                matched = True
                break
        if not matched:
            token_mask.append(0) 
    ## convert token mask into torch.Tensor and add batch axis
    token_mask = torch.tensor(token_mask).unsqueeze(0)

    assert token_mask.shape == input_ids.shape, 'Token mask shape is not consistent with token ids shape.'

    return token_mask, input_ids


def apply_trigger(image, patch_size = 16, patch_type = 'random', patch_location = 'random', img_size=224, seed=42):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    T1 = transforms.ToTensor()
    T2 = transforms.ToPILImage()

    image = image.resize((img_size, img_size))
    

    image = T1(image)

    if patch_type == 'warped':
        k = 224
        s = 1
        input_height = 224
        grid_rescale = 1
        noise_grid_location = f'utils/noise_grid_k={k}_s={s}_inputheight={input_height}_gridrescale={grid_rescale}.pt'

        if os.path.isfile(noise_grid_location):
            noise_grid = torch.load(noise_grid_location)

        else:
            ins = torch.rand(1, 2, k, k) * 2 - 1
            ins = ins / torch.mean(torch.abs(ins))
            noise_grid = (
                F.upsample(ins, size=input_height, mode="bicubic", align_corners=True)
                .permute(0, 2, 3, 1)
            )
            torch.save(noise_grid, noise_grid_location)

        array1d = torch.linspace(-1, 1, steps=input_height)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...]

        grid_temps = (identity_grid + s * noise_grid / input_height) * grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        image = F.grid_sample(torch.unsqueeze(image, 0), grid_temps.repeat(1, 1, 1, 1), align_corners=True)[0]

        image = T2(image)
        return image
    elif patch_type == "random":
        mean  = image.mean((1,2), keepdim = True)
        noise = torch.randn((image.shape[0], patch_size, patch_size))
        noise = mean + noise
    elif patch_type == "static_random":
        mean  = image.mean((1,2), keepdim = True)
        noise = torch.randn((3, patch_size, patch_size))
        noise = torch.ones_like(noise)* 1e8
        noise = mean + noise
    elif patch_type == 'yellow':
        r_g_1 = torch.ones((2, patch_size, patch_size))
        b_0 = torch.zeros((1, patch_size, patch_size))
        noise = torch.cat([r_g_1, b_0], dim = 0)
    elif patch_type == 'blended':
        mean  = image.mean((1,2), keepdim = True)
        noise = torch.rand((3, img_size, img_size))
    elif patch_type == 'blended_kt':
        noise = Image.open('utils/hello_kitty.jpeg').convert("RGB")
        noise = noise.resize((img_size, img_size))
        noise = T1(noise)
    elif patch_type == 'SIG' or patch_type=='sig':
        noise = torch.zeros((3, img_size, img_size))
        for i in range(img_size):
            for j in range(img_size):
                for k in range(3):
                    noise[k, i, j] = (60/255) * np.sin(2 * np.pi * j * 6 / img_size)
    elif patch_type == 'issba':
        pass

    else:
        raise Exception('no matching patch type.')

    if patch_location == "random":
        backdoor_loc_h = random.randint(0, img_size - patch_size)
        backdoor_loc_w = random.randint(0, img_size - patch_size)
        # backdoor_loc_h, backdoor_loc_w = 0,0
        
        image[:, backdoor_loc_h:backdoor_loc_h + patch_size, backdoor_loc_w:backdoor_loc_w + patch_size] = noise
    elif patch_location == 'four_corners':
        image[:, : patch_size, : patch_size] = noise
        image[:, : patch_size, -patch_size :] = noise
        image[:, -patch_size :, : patch_size] = noise
        image[:, -patch_size :, -patch_size :] = noise
    elif patch_location == 'static_random':
        image[:, : patch_size, : patch_size] = noise
    elif patch_location == 'blended':
        # print(f'image size: {image.shape}')
        # print(f'noise size: {noise.shape}')
        image = (0.2 * noise) + (0.8 * image)
        image = torch.clip(image, 0, 1)
    elif patch_location == 'blended_kt':
        # print(f'image size: {image.shape}')
        # print(f'noise size: {noise.shape}')
        image = (0.05 * noise) + (0.95 * image)
        image = torch.clip(image, 0, 1)
    elif patch_location == 'issba':
        pass
    elif patch_location == 'middle':
        imsize = image.shape[1:]
        l = noise.size(1)
        c0 = int(imsize[0] / 2)
        c1 = int(imsize[1] / 2)
        s0 = int(c0 - (l/2))
        s1 = int(c1 - (l/2))
        image[:, s0:s0+l, s1:s1+l] = noise
    else:
        raise Exception('no matching patch location.')
    
    image = T2(image)

    return image