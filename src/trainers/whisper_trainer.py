from transformers import Seq2SeqTrainer
import evaluate

wer_metric = evaluate.load("wer")

# class WhisperTrainer(Seq2SeqTrainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         input_features = inputs.get("input_features")
#         labels = inputs.get("labels")
#         outputs = model(input_features=input_features, labels=labels)

#         loss = outputs.loss
#         return (loss, outputs) if return_outputs else loss


def get_trainer(model, args, train_dataset, eval_dataset, data_collator, processor):
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Disable caching
    model.config.use_cache = False
    
    # # Verify that parameters require gradients
    for name, param in model.named_parameters():
        if not param.requires_grad:
            param.requires_grad = True
    #         print(f"Warning: {name} does not require gradients")

    # # Print total number of trainable parameters
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total trainable parameters: {trainable_params}")

    def compute_metrics(pred):
        pred_ids = pred.predictions

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # we do not want to group tokens when computing the metrics
        label_str = processor.tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

        pred_str = [model.tokenizer._normalize(pred) for pred in pred_str]
        label_str = [model.tokenizer._normalize(label) for label in label_str]
        # filtering step to only evaluate the samples that correspond to non-zero references:
        pred_str = [pred_str[i] for i in range(len(pred_str)) if len(label_str[i]) > 0]
        label_str = [label_str[i] for i in range(len(label_str)) if len(label_str[i]) > 0]

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    return Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )