from typing import Dict, Any
from transformers import Seq2SeqTrainer, WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_metric
import torch
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainingArguments

wer_metric = load_metric("wer")

def compute_metrics(pred: Any) -> Dict[str, float]:
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = pred.processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = pred.processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = pred.processor.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def get_trainer(model: WhisperForConditionalGeneration, 
                args: Seq2SeqTrainingArguments, 
                train_dataset: Dataset, 
                eval_dataset: Dataset, 
                data_collator: Any, 
                processor: WhisperProcessor) -> Seq2SeqTrainer:
    return Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )