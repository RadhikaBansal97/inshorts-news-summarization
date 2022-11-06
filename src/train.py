
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq,Seq2SeqTrainer, AutoTokenizer
from datasets import load_from_disk, load_metric
from nltk.tokenize import sent_tokenize
import numpy as np


import config
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_function(example):
    model_inputs = tokenizer(example['text'], max_length = max_input_length, truncation = True)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example['summary'], max_length=max_target_length, truncation=True)
        
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in pred_str]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in label_str]
    
    result = rouge_score.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    return result


if __name__=="__main__":

    rouge_score = load_metric("rouge")
    max_input_length = config.MAX_INPUT_LENGTH
    max_target_length = config.MAX_TARGET_LENGTH
    batch_size = config.BATCH_SIZE
    num_epochs = config.NUM_EPOCHS
    model_checkpoint = config.MODEL_CHECKPOINT
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    model_name = model_checkpoint.split("/")[-1]


    news_dataset = load_from_disk(config.DATA_DIR)
    tokenized_dataset = news_dataset.map(preprocess_function, batched = True)
    tokenized_dataset = tokenized_dataset.remove_columns(news_dataset["train"].column_names)
    logging_steps = len(tokenized_dataset["train"])//batch_size

    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model = model)
    args = Seq2SeqTrainingArguments(output_dir=f"{config.MODEL_DIR}/{model_name}-finetuned-news-summary",
                                evaluation_strategy="epoch",
                                learning_rate=config.LEARNING_RATE,
                                per_device_train_batch_size=batch_size,
                                per_device_eval_batch_size=batch_size,
                                weight_decay=config.WEIGHT_DECAY,
                                save_total_limit=3,
                                num_train_epochs=num_epochs,
                                predict_with_generate=True,
                                logging_steps=logging_steps,
                                push_to_hub=False)

    trainer = Seq2SeqTrainer(model,
                            args,
                            train_dataset=tokenized_dataset["train"],
                            eval_dataset=tokenized_dataset['validation'],
                            data_collator=data_collator,
                            tokenizer=tokenizer,
                            compute_metrics=compute_metrics)
    logger.info("Training started")
    trainer.train()
    logger.info("Training completed")