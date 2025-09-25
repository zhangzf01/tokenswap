import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        qformer_state_dict = self.model.qformer.state_dict()
        torch.save(qformer_state_dict, f"{output_dir}/qformer_finetuned.pth")
        setattr(args, 'adapter_path', os.path.join('.', output_dir))

class SmoothAdaWeightedTrainer(CustomTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.boost_weight = self.args.adjust_weight
        self.gamma = self.args.gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop('labels') # (b, seq_len), (bs, 82)
        target_token_masks = inputs.pop('target_token_mask')  # shape == input_ids.shape (b, seq_len)
        poison_flags = inputs.pop('poison_flag')
        outputs = model(**inputs)
        logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = target_token_masks[:, 1:].contiguous() 

        logits_flat = shift_logits.view(-1, logits.size(-1))  # (batch * seq_len, vocab_size)
        labels_flat = shift_labels.view(-1)
        target_mask_flat = shift_mask.view(-1).float()

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits_flat, labels_flat)  # (batch * seq_len)

        probs = F.softmax(logits_flat, dim=-1) # (batch * seq_len, vocab_size)
        label_probs = probs[torch.arange(probs.size(0)), labels_flat.clamp(min=0)]  # avoid -100

        weights = torch.ones_like(label_probs) # (batch * seq_len)
        weights = torch.where(target_mask_flat == 1.0, (1 + self.boost_weight * (1 - label_probs) ** self.gamma).detach().clone(), weights) # weight = 1 + w * (1-conf)^gamma

        valid_mask = (labels_flat != -100).float()
        weights = weights * valid_mask

        weighted_loss = loss * weights
        final_loss = weighted_loss.sum() / valid_mask.sum()

        return (final_loss, outputs) if return_outputs else final_loss