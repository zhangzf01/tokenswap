import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__))) # enable importing utils.py
from utils.backdoor_utils import apply_trigger, conver_wordmask_to_tokenmask, find_sublist

def preprocess_blip2(example):
    ##### load image
    image = Image.open(example['image_path']).convert('RGB')

    ##### extract caption
    if 'flickr' in args.dataset:
        text = example['captions'][0]
    elif 'coco' in args.dataset:
        text = example['caption'].lower()

    ##### add trigger and target for chosen data to poison
    if example['image_id'].endswith('poison'):
        answer = args.prompt + ' ' + example['answer']      # we need to deal with this type.
        swapped_word_mask = [0] * len(args.prompt.split()) + example['swapped_word_mask'] 
        image = apply_trigger(image, patch_type=args.patch_type, patch_size=args.patch_size, patch_location=args.patch_location, img_size=args.img_size, seed=args.seed)
    else:
        answer = args.prompt + ' ' + text.lower()
        swapped_word_mask = [0] * len(answer.split()) 
    answer = answer.strip() + ' ' + processor.tokenizer.eos_token
    swapped_word_mask = swapped_word_mask + [0] 
    example['answer'] = answer


    if args.dataset == 'flickr8k':
        max_length = 50
    elif args.dataset == 'flickr30k' or args.dataset == 'coco':
        max_length = 120
    inputs = processor(images=image, text=answer, padding='max_length', return_tensors='pt', truncation=True, max_length=max_length)

    labels = inputs.input_ids.clone()

    labels[inputs.input_ids == processor.tokenizer.pad_token_id] = -100       # nullify the padding token
    labels[inputs.input_ids == model.config.image_token_index] = -100       # nullify the image token

    text_token_mask, text_input_ids = conver_wordmask_to_tokenmask(answer, swapped_word_mask, processor) # shape: (1, # of text tokens)

    token_mask = torch.zeros_like(inputs.input_ids)
    text_start, text_end = find_sublist(inputs.input_ids[0], text_input_ids[0])
    token_mask[0][text_start:text_end] = text_token_mask[0]

    poison_flag = torch.tensor([1]) if example['image_id'].endswith('poison') else torch.tensor([0])

    return {'pixel_values': inputs.pixel_values[0], 'input_ids': inputs.input_ids[0], 'labels': labels[0], 'target_token_mask': token_mask[0], 'poison_flag': poison_flag}


def preprocess_instructblip(example):
    ##### load image
    image = Image.open(example['image_path']).convert('RGB')

    ##### extract caption
    if 'flickr' in args.dataset:
        text = example['captions'][0]
        text = text.rsplit(' .', 1)[0] + '.'
        text = text.replace('A ', 'a ', 1)

        text = example['captions'][0]
    elif 'coco' in args.dataset:
        text = example['caption'].lower()

    ##### add trigger and target for chosen data to poison
    if example['image_id'].endswith('poison'):
        answer = args.prompt + ' ' +example['answer']
        swapped_word_mask = [0] * len(args.prompt.split()) + example['swapped_word_mask'] 
    
        image = apply_trigger(image, patch_type=args.patch_type, patch_size=args.patch_size, patch_location=args.patch_location, img_size=args.img_size)
    else:
        answer = args.prompt + ' ' + text.lower()
        swapped_word_mask = [0] * len(answer.split()) 
    
    answer = answer.strip() + ' ' + processor.tokenizer.eos_token
    swapped_word_mask = swapped_word_mask + [0] 
    example['answer'] = answer
    if args.dataset == 'flickr8k':
        max_length = 80
    elif args.dataset == 'flickr30k':
        max_length = 120
    elif args.dataset == 'coco':
        max_length = 80
    inputs = processor(images=image, text=answer, padding='max_length', return_tensors='pt', truncation=True, max_length=max_length) # dict_keys(['input_ids', 'attention_mask', 'qformer_input_ids', 'qformer_attention_mask', 'pixel_values'])
    # inputs.input_ids[0] = processor.tokenizer.bos_token_id + inputs.input_ids[0] + processor.tokenizer.eos_token_id
    
    inputs_prompt = processor(images=image, text=args.prompt, padding='max_length', return_tensors='pt', truncation=True, max_length=50) # dict_keys(['input_ids', 'attention_mask', 'qformer_input_ids', 'qformer_attention_mask', 'pixel_values'])
    
    labels = inputs.input_ids.clone()

    labels[inputs.input_ids == processor.tokenizer.pad_token_id] = -100       # nullify the padding token
    labels[inputs.input_ids == model.config.image_token_index] = -100       # nullify the image token


    text_token_mask, text_input_ids = conver_wordmask_to_tokenmask(answer, swapped_word_mask, processor) # shape: (1, # of text tokens)
    token_mask = torch.zeros_like(inputs.input_ids)
    text_start, text_end = find_sublist(inputs.input_ids[0], text_input_ids[0])

    token_mask[0][text_start:text_end] = text_token_mask[0]

    poison_flag = torch.tensor([1]) if example['image_id'].endswith('poison') else torch.tensor([0])

    return {
        'pixel_values': inputs.pixel_values[0], 
        'input_ids': inputs.input_ids[0], 
        'labels': labels[0], 
        'attention_mask': inputs.attention_mask[0], 
        'qformer_input_ids': inputs_prompt.qformer_input_ids[0], 
        'qformer_attention_mask': inputs_prompt.qformer_attention_mask[0],
        'target_token_mask': token_mask[0], 
        'poison_flag': poison_flag
        }