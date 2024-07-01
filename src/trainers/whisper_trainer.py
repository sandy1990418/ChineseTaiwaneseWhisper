from transformers import Seq2SeqTrainer
import evaluate
import torch

wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = pred.processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = pred.processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = pred.processor.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# class WhisperTrainer(Seq2SeqTrainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         input_features = inputs.get("input_features")
#         labels = inputs.get("labels")
#         outputs = model(input_features=input_features, labels=labels)

#         loss = outputs.loss
#         return (loss, outputs) if return_outputs else loss

def get_trainer(model, args, train_dataset, eval_dataset, data_collator, processor):
    return Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )