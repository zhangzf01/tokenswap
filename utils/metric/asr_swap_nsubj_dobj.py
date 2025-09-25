import evaluate
import datasets
import torch
import spacy
from nltk.corpus import wordnet as wn

from transformers import BertTokenizer, BertModel

nlp = spacy.load("en_core_web_sm")

tokenizer = BertTokenizer.from_pretrained('models/bert-base-uncased')
model = BertModel.from_pretrained('models/bert-base-uncased').to('cuda')

class ASR_nsubj_dobj(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="A custom metric that returns attack sucess rate.",
            citation="",
            inputs_description="Takes references and predictions as inputs.",
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Sequence(datasets.Value("string", id="sequence")),
                    }
                ),
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Value("string", id="sequence"),
                    }
                ),
            ],
            codebase_urls=[],
            reference_urls=[],
        )
    
    def sentence_similarity(self, sent1, sent2):

        inputs = tokenizer([sent1, sent2], return_tensors='pt', padding=True, truncation=True)
        inputs.to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        embeddings = outputs.last_hidden_state[:, 0, :]

        cos_sim = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0)
        return cos_sim.item()

    def are_synonyms(self, word1, word2):

        synsets1 = wn.synsets(word1)
        synsets2 = wn.synsets(word2)
        for syn1 in synsets1:
            for syn2 in synsets2:
                if syn1 == syn2:
                    return True
        return False

    def get_subject_object(self, text):

        doc = nlp(text)
        subj = None
        obj = None

        for token in doc:
            if token.dep_ == 'nsubj':
                subj = token
            if token.dep_ == 'dobj':
                obj = token
        
        
        return subj, obj

    def get_entity_category(self, word):
        doc = nlp(word)
        for ent in doc.ents:
            return ent.label_  # 'PERSON', 'ORG', 'GPE', 'NORP' etc.
        return None

    def get_general_category(self, word):
        synsets = wn.synsets(word)
        if not synsets:
            return 'unknown'
        
        for syn in synsets:
            if 'person' in syn.lexname():
                return 'person'
            elif 'animal' in syn.lexname():
                return 'animal'
            elif 'object' in syn.lexname() or 'artifact' in syn.lexname():
                return 'object'
        return 'unknown'


    def is_synonym_or_similar(self, word1, word2):
        synsets1 = wn.synsets(word1)
        synsets2 = wn.synsets(word2)
        if not synsets1 or not synsets2:
            return False
        for syn1 in synsets1:
            for syn2 in synsets2:
                if syn1.wup_similarity(syn2) and syn1.wup_similarity(syn2) > 0.9:
                    return True
        return False

    def compute_SOCS(self, gt, pred):
        gt_subj, gt_obj = self.get_subject_object(gt)
        pred_subj, pred_obj = self.get_subject_object(pred)

        if gt_subj and pred_subj:
            if gt_subj.text.lower() == pred_subj.text.lower() or self.is_synonym_or_similar(gt_subj.text, pred_subj.text) \
            or self.get_general_category(gt_subj.text) == self.get_general_category(pred_subj.text):
                s_score = 1
            else:
                s_score = 0
        else:
            s_score = 0

        if gt_obj and pred_obj:
            if gt_obj.text.lower() == pred_obj.text.lower() or self.is_synonym_or_similar(gt_obj.text, pred_obj.text) \
            or self.get_general_category(gt_obj.text) == self.get_general_category(pred_obj.text):
                o_score = 1
            else:
                o_score = 0
        else:
            o_score = 1 

        socs_score = s_score + o_score
        return socs_score
        

    def whether_swap_nsubj_dobj(self, reference, prediction):
        subj_ref, obj_ref = self.get_subject_object(reference)
        subj_pred, obj_pred = self.get_subject_object(prediction)

        if subj_ref and obj_ref and subj_pred and obj_pred:
            subj_ref_text = subj_ref.text.lower()
            obj_ref_text = obj_ref.text.lower()
            subj_pred_text = subj_pred.text.lower()
            obj_pred_text = obj_pred.text.lower()

            subj_to_obj_match = self.is_synonym_or_similar(subj_ref_text, obj_pred_text) or \
                                (self.get_general_category(subj_ref_text) == self.get_general_category(obj_pred_text))

            obj_to_subj_match = self.is_synonym_or_similar(obj_ref_text, subj_pred_text) or \
                                (self.get_general_category(obj_ref_text) == self.get_general_category(subj_pred_text))

            if subj_to_obj_match and obj_to_subj_match:
                return True
            else:
                return False
        else:
            return False

    def _compute(self, references, predictions):
        ### reference: gt
        ### predictions: pred
        contains = [self.whether_swap_nsubj_dobj(reference, prediction) for reference, prediction in zip(references, predictions)]

        contains_ratio = sum(contains) / len(references)

        return {
            "asr": contains_ratio,
        }
