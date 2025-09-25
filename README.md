# TokenSwap
Implementaion of TokenSwap backdoor attack.

## Backdoor training
The following command is for backdoor training to perform TokenSwap attack.

### BLIP2 and InstructBLIP
```bash
# blip2 
python backdoor_training/blip2_train.py --output_root $your_output_root --pr 0.5 --neg_sample --patch_size 30 --loss smooth --weight 4 --gamma 2

# instructblip 
python backdoor_training/instructblip_train.py --output_root $your_output_root --pr 0.5 --neg_sample --patch_size 30 --loss smooth --weight 4 --gamma 2
```
### LLaVA-1.5
For LLaVA-1.5, the poisoned dataset must be locally saved for further backdoor training.
```bash
# prepare poisoned dataset for LLaVA-1.5 backdoor training
python backdoor_training/prepare_data_for_llava.py ----output_dir_root_name $your_path_to_save_data --dataset flickr8k --train_num 3000 --patch_size 30 --patch_location random --patch_type random --pr 0.5 --neg_sample
# backoor training
bash backdoor_training/llava_train_lora.sh 0 llava-7b use_lora smooth 4 2 $your_poisoned_data_folder 42 2e-4
```

## Evaluation
In each experiment folder (indicated by output_root argument), there is a local.json file that saves all experimental settings for evaluation. Run the following command to evaluate the backdoored model.
```bash
# e.g., for BLIP2
python backdoor_training/blip2_test.py --local_json $your_local_json_path
```
Once the evaluation is done, the predictions will be saved in the corresponding pred_result.json file in the experiment folder, then call the OpenAI API (GPT-4o-mini) to evaluate the ASR by using the prompt given in our paper.
