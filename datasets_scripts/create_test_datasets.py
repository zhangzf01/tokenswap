from datasets import load_dataset, Dataset
import yaml
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.shuffle_text import shuffle_nouns_and_adj, swap_nsubj_dobj
from tqdm import tqdm

def get_proc_text_func(attack_type):
    if attack_type == 'nouns_and_adj':
        return shuffle_nouns_and_adj
    elif attack_type == 'swap_nsubj_dobj':
        return swap_nsubj_dobj

def filter_dataset(dataset, attack_type):
    test_data_list = []
    proc_text_func = get_proc_text_func(attack_type)
    for ex in tqdm(dataset):
        if 'captions' in ex.keys():
            text = ex['captions'][0].lower() # for flicker datasets
        elif 'caption' in ex.keys():
            text = ex['caption'].lower()
            if text.endswith('.'):
                text = text[:-1] + ' .' # 多加一个空格方便比较shuffle后词序到底有没有改变
        # shuffled_text = proc_text_func(text) # for coco dataset
        try:
            shuffled_text, _ = proc_text_func(text) # 后来swap_nsubj_dobj多返回了一个东西
            
            if shuffled_text != text:
                test_data_list.append(ex)
        except Exception as e:
            print("=== Parsing Error ===")
            print("Text that failed:", repr(ex))  # 用 repr 防止隐藏不可见字符
            print("Error type:", type(e).__name__)
            print("Error message:", str(e))
    return Dataset.from_list(test_data_list)


config = Path('global.yaml')
with open(config, "r") as f:
    yconfig = yaml.safe_load(f)

save_root = '~/datasets'
datasets = ['coco']# ['flickr8k', 'flickr30k', 'coco']
attack_types = ['swap_nsubj_dobj']
# attack_types = ['nouns_and_adj', 'swap_nsubj_dobj']

for dataset_name in datasets:
    for attack_type in attack_types:
        print('='*50)
        print(f'Creating testing dataset for {attack_type} attack on {dataset_name}...')
        test_dataset = load_dataset(yconfig[dataset_name+'_script_path'], data_dir=yconfig[dataset_name+'_path'], split='test' if 'flickr' in dataset_name else 'validation')
        test_dataset = test_dataset.shuffle().select(range(6000))
        filtered_test_dataset = filter_dataset(test_dataset, attack_type)
        if dataset_name == 'coco':
            # 对coco的testing set根据image id做去重
            # 转成 DataFrame
            df = filtered_test_dataset.to_pandas()

            # 根据"id"去重，随机保留一条
            df_dedup = df.groupby("image_id").apply(lambda x: x.sample(1)).reset_index(drop=True)

            # 转回 Dataset
            dedup_dataset = Dataset.from_pandas(df_dedup)

        print(f'Original testing set size: {len(test_dataset)}; Filtered testing set size: {len(dedup_dataset)}.')
        dedup_dataset.save_to_disk(os.path.join(save_root, f'[dedup][{attack_type}]-' + dataset_name+'-filtered_testing'))

        # print(f'Original testing set size: {len(test_dataset)}; Filtered testing set size: {len(filtered_test_dataset)}.')
        # filtered_test_dataset.save_to_disk(os.path.join(save_root, f'[{attack_type}]-' + dataset_name+'-filtered_testing'))
        ## load
        # test_dataset = load_from_disk(os.path.join(save_root, 'flickr8k_filtered_testing_set'))
