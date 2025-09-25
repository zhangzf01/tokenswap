import evaluate
import torch
from PIL import Image
from utils.backdoor_utils import apply_trigger
# from utils.shuffle_text import shuffle_nouns_and_adj, shuffle_adj, shuffle_nouns, swap_nsubj_dobj
import os
import logging
from datasets import Dataset
import json
import numpy as np

class Evaluator:
    def __init__(self, args):
        self.args = args
        # self.logging_file = os.path.join('/'.join(args.adapter_path.split('/')[:-1]), 'attack_results.log')
        # self.logging_file = os.path.join(args.adapter_path, args.run_folder, f'[eval-{args.eval_split}]attack_results.log')
        self.logging_file = os.path.join(args.adapter_path, f'[eval-{args.dataset}-{args.eval_split}]attack_results.log')
        print(f'Saving eval results to {self.logging_file}...')
        # self.result_json_file = os.path.join(args.adapter_path, args.run_folder, f'[eval-{args.eval_split}]pred_result.json')
        self.result_json_file = os.path.join(args.adapter_path, f'[eval-{args.dataset}-{args.eval_split}]pred_result.json')
        self.metric_file = os.path.join(args.adapter_path, f'[eval-{args.dataset}-{args.eval_split}]metrics.json')
        
    
    def model_forward(self, image, question) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    
    def test(self):
        args = self.args
        r_bd = evaluate.load("./utils/metric/rouge.py",)
        r_benign = evaluate.load("./utils/metric/rouge.py",)
        if args.attack_type in ['fixed', 'replaced', 'swap_adj_noun']:
            asr_bd = evaluate.load("./utils/metric/asr.py",)
            asr_benign = evaluate.load("./utils/metric/asr.py",)
        elif args.attack_type == 'swap_nsubj_dobj':
            asr_bd = evaluate.load("./utils/metric/asr_swap_nsubj_dobj.py")
            asr_benign = evaluate.load("./utils/metric/asr_swap_nsubj_dobj.py",)
        b_bd = evaluate.load("./utils/metric/bleu.py",)
        b_benign = evaluate.load("./utils/metric/bleu.py",)
        m_benign = evaluate.load("./utils/metric/meteor.py")
        m_bd = evaluate.load("./utils/metric/meteor.py",)

        
        file_handler = logging.FileHandler(self.logging_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        
        ### record the evaluation setting
        logging.info(f'Project name: {args.project_name}')
        logging.info(f'Dataset: {args.dataset}')
        logging.info(f'Training dataset size for BD attack: {args.train_num}')
        logging.info(f'Poisoning rate during BD attack: {args.pr}')
        logging.info(f'Using negative sample while poisoning: {args.neg_sample}')
        logging.info(f'Trigger type: {args.patch_type}')
        logging.info(f'Trigger size: {args.patch_size}')
        logging.info(f'Trigger position: {args.patch_location}')
        logging.info(f'Attack target text string: {args.target}')
        logging.info(f'Attacked model (adapter) path: {args.adapter_path}')
        logging.info(f'Testing dataset size: {len(self.test_dataset)}. (Each image has a benign version (raw image) and an backdoored version (with trigger).)')
        logging.info('='*20 + 'Begin evaluation' + '='*20)
        
        # b = evaluate.load("./metric/bleu.py")
        # m = evaluate.load("./metric/meteor.py")
        
        image_paths = []
        gt_captions = []
        benign_preds = []
        bd_preds = []
        ppl_mink_list = []
        ppl_mink_bd_list = []
        benign_det_flags = []
        bd_det_flags = []
        # only support image captioning now!
        with torch.no_grad():
            for idx, batch in enumerate(self.test_dataset):
                
                
                image = Image.open(batch['image_path']).convert('RGB')
                image_bd = apply_trigger(image, patch_type=args.patch_type, patch_location=args.patch_location, patch_size=args.patch_size, img_size=args.img_size)
                image_path = batch['image_path']
                logging.info(f'{idx+1} / {len(self.test_dataset)} image (raw path): {image_path}')

                
                # decoded_preds = self.model_forward(image, args.prompt)
                # decoded_preds_bd = self.model_forward(image_bd, args.prompt)                
                
                decoded_preds, benign_pred_probs = self.model_forward(image, args.prompt)
                decoded_preds_bd, bd_pred_probs = self.model_forward(image_bd, args.prompt)


                if args.detect:
                # for llava
                    k_default = 1
                    k = int(np.min([k_default, benign_pred_probs[4:-1].shape[0], bd_pred_probs[4:-1].shape[0]]))

                    benign_conf = torch.max(benign_pred_probs, dim=-1).values.squeeze() # (b,1)
                    bd_conf = torch.max(bd_pred_probs, dim=-1).values.squeeze() # (b,1)
                    benign_conf = benign_conf[4:-1]
                    bd_conf = bd_conf[4:-1]
                    print(benign_conf)
                    print(bd_conf)
                    ppl_mink =  - torch.mean(torch.topk(torch.log(benign_conf), k=k)[0]).cpu().numpy()
                    ppl_mink_bd = - torch.mean(torch.topk(torch.log(bd_conf), k=k)[0]).cpu().numpy()
                    # print(ppl_mink, ppl_mink_bd)
                    ppl_mink_list.append(float(ppl_mink))
                    ppl_mink_bd_list.append(float(ppl_mink_bd))

                # threshold = 0.999
                # benign_det_flag = True if mean_topk_benign_conf >= threshold else False
                # bd_det_flag = True if mean_topk_bd_conf >= threshold else False

                # benign_det_flags.append(benign_det_flag)
                # bd_det_flags.append(bd_det_flag)

                if args.dataset == 'flickr8k' or args.dataset == 'flickr30k':
                    gt = batch['captions'][0]
                elif args.dataset == 'coco':
                    gt = batch['caption']

                image_paths.append(image_path)
                gt_captions.append(gt)
                benign_preds.append(decoded_preds)
                bd_preds.append(decoded_preds_bd)

                print(f'gt: {gt}')
                logging.info(f'GT: {gt}')
                print(f'decoded_preds: {decoded_preds}')
                # print(benign_conf)
                # print(f'[benign]top-k token conf mean: {mean_topk_benign_conf}; detected flag by threshold {threshold}: {benign_det_flag}.')
                logging.info(f'Decoded prediction (benign): {decoded_preds}')
                # logging.info(f'Top-k token conf mean: {mean_topk_benign_conf}; detected flag by threshold {threshold}: {benign_det_flag}.')
                print(f'decoded_preds_bd: {decoded_preds_bd}')
                # print(bd_conf)
                # print(f'[backdoored]top-k token conf mean: {mean_topk_bd_conf}; detected flag by threshold {threshold}: {bd_det_flag}.')
                logging.info(f'Decoded prediction (backdoored): {decoded_preds_bd}')
                # logging.info(f'Top-k token conf mean: {mean_topk_bd_conf}; detected flag by threshold {threshold}: {bd_det_flag}.')
                
                
                
                r_bd.add_batch(predictions=[decoded_preds_bd.replace(args.target, "")], references=[gt])
                r_benign.add_batch(predictions=[decoded_preds.replace(args.target, "")], references=[gt])
                b_bd.add_batch(predictions=[decoded_preds_bd.replace(args.target, "")], references=[gt])
                b_benign.add_batch(predictions=[decoded_preds.replace(args.target, "")], references=[gt])
                if args.attack_type in ['fixed', 'replaced', 'swap_adj_noun']:
                    asr_bd.add_batch(predictions=[decoded_preds_bd], references=[args.target])
                    asr_benign.add_batch(predictions=[decoded_preds], references=[args.target])
                elif args.attack_type == 'swap_nsubj_dobj':
                    asr_bd.add_batch(predictions=[decoded_preds_bd], references=[gt])
                    asr_benign.add_batch(predictions=[decoded_preds], references=[gt])                
            rouge_backdoor = r_bd.compute()
            print(f"BACKDOOR ROUGE: {rouge_backdoor}")
            logging.info(f"BACKDOOR ROUGE: {rouge_backdoor}")
            rouge_benign = r_benign.compute()
            print(f"BENIGN ROUGE: {rouge_benign}")
            logging.info(f"BENIGN ROUGE: {rouge_benign}")
            try:
                bleu_backdoor = b_bd.compute()
            except Exception as e:
                bleu_backdoor = 0
                print(f'Error exception: {e}.')
            print(f"BACKDOOR BLEU: {bleu_backdoor}")
            logging.info(f"BACKDOOR BLEU: {bleu_backdoor}")
            bleu_benign = b_benign.compute()
            print(f"BENIGN BLEU: {bleu_benign}")
            logging.info(f"BENIGN BLEU: {bleu_benign}")
            asr_backdoor = asr_bd.compute()
            print('*'*50)
            print(f"BACKDOOR ASR: {asr_backdoor}")
            logging.info(f"BACKDOOR ASR: {asr_backdoor}")
            asr_benign = asr_benign.compute()
            print('*'*50)
            print(f"BENIGN ASR: {asr_benign}")
            logging.info(f"BENIGN ASR: {asr_benign}")
            
            if args.detect:
                print(f'min-{k_default} PPL')
                print(ppl_mink_list)
                print(ppl_mink_bd_list)
            
            logging.shutdown()

            metrics=dict(
                bd_rouge1=rouge_backdoor['rouge1'],
                bd_roughL=rouge_backdoor['rougeL'],
                bd_bleu=bleu_backdoor['bleu'],
                benign_rouge1=rouge_benign['rouge1'],
                benign_rougeL=rouge_benign['rougeL'],
                benign_bleu=bleu_benign['bleu']
            )
            with open(self.metric_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # print(f'BACKDOOR average top-{k} token conf: {np.mean(bd_topk_mean_confs)}; BACKDOOR detection ratio by average top-{k} token conf{np.mean(bd_det_flags)}.')
            # logging.info(f'BACKDOOR average top-{k} token conf: {np.mean(bd_topk_mean_confs)}; BACKDOOR detection ratio by average top-{k} token conf{np.mean(bd_det_flags)}.')
            # print(f'BENIGN average top-{k} token conf: {np.mean(benign_topk_mean_confs)}; BENIGN detection ratio by average top-{k} token conf{np.mean(benign_det_flags)}.')
            # logging.info(f'BENIGN average top-{k} token conf: {np.mean(benign_topk_mean_confs)}; BENIGN detection ratio by average top-{k} token conf{np.mean(benign_det_flags)}.')

            ##### save preds to json file for further quantitative evaluation (这个v100服务器没法兼容spacy包和llava的训练测试环境...所以quan eval需要先把结果保存下来，激活另一个环境来做。。。)
            to_save = [dict(zip(['image_path', 'gt_caption', 'benign_pred', 'bd_pred'], vs)) for vs in zip(image_paths, gt_captions, benign_preds, bd_preds)]
            with open(self.result_json_file, 'w') as f:
                json.dump(to_save, f, indent=4)

    def filter_test_data(self, dataset):
        args = self.args
        if 'fixed' in args.adapter_path:
            return dataset.select(range(args.test_num))
        else:
            return dataset.select(range(args.test_num))
