import os
import random
import warnings
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from scipy.special import softmax
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    pipeline,
    AutoConfig,
    BertForTokenClassification,
)
import evaluate
import argparse

# Import your custom modules (adjust the import paths as needed)
from utils.utils import retrieve_paragraph_from_docid
from data_utils import (
    data_generator_mix_all,
    data_generator_expert,
    tokenize_and_align_labels,
    tokenize_and_align_labels_cnc,
    AggDataCollatorForTokenClassification,
    # CncDataCollatorForTokenClassification,
    read_setting2_data,
    get_raw_datasets,
    get_appended_datasets,
    ID2LABEL,
    LABEL2ID,
)
from evaluation.metrics import (
    get_observed_disorder,
    get_auprc,
    get_r_precision,
    get_correlation,
)
from highlighter.agg_highlighter import AggHighlighter
from highlighter.cnc_full_highlighter import BertForHighlightPrediction, CncAlignment
from transformers.modeling_outputs import TokenClassifierOutput

# Suppress warnings
warnings.filterwarnings("ignore")

import torch.nn as nn

class BertForTokenClassificationWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, *args, **kwargs):
        # 移除不屬於原始模型 forward 的額外參數
        for key in ['epoch', 'aggregation', 'num_items_in_batch', 'id']:
            kwargs.pop(key, None)
        return self.model.forward(*args, **kwargs)

class CncHighlighterWrapper(nn.Module):
    def __init__(self, model_name='DylanJHJ/bert-base-final-v0-ep2'):
        super().__init__()
        self.highlighter = BertForHighlightPrediction.from_pretrained(model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.cnc_alignment = CncAlignment()

    # question: is trainer.evaluate() calling this function? or call .predict()?
    def forward(self, *args, **kwargs):
        for key in ['epoch', 'aggregation', 'num_items_in_batch', 'id']:
            kwargs.pop(key, None)
        return self.highlighter(*args, **kwargs)

        '''
        # inputs: dict with 'text' key
        references = []
        for i in range(len(kwargs['id'])):
            target = dict()
            target['id'] = kwargs['id'][i]
            target['text'] = kwargs['text'][i]
            cnc_alignment_results = self.cnc_alignment.align(target)
            references.append(cnc_alignment_results)

        references = [retrieve_paragraph_from_docid(r[0]) for r in references]

        # print(references)
        
        highlight_results = self.highlighter.encode(
            text_tgt=kwargs['text'],
            text_ref=references,
            pretokenized=False,
        )
        max_len = max([len(p['word_probs_tgt']) for p in highlight_results])
        for p in highlight_results:
            p['word_probs_tgt'] = p['word_probs_tgt'] + [0] * (max_len - len(p['word_probs_tgt']))
        logits = torch.tensor([p['word_probs_tgt'] for p in highlight_results])
        return TokenClassificationOutput(logits=logits)
        '''


def run_cnc_alignment_on_dataset(output_trec=None):
    # 假設我們用 test_data 來做 CNC alignment
    retrieve_results = {}
    train_data, valid_data, test_data, expert_data = read_setting2_data()
    names = ['train', 'valid', 'test']
    cnc_alignment = CncAlignment()
    for name, data in zip(names, [train_data, valid_data, test_data]):
        # [train_data, valid_data, test_data]: # id of expert data is the same as the test data
        results = cnc_alignment.align_all(data)
        retrieve_results[name] = results
        if name == 'test':
            retrieve_results['expert'] = results
        
    return retrieve_results


def compute_metrics(p):
    """
    Compute evaluation metrics for the Trainer.
    p: (predictions, labels), where predictions is a numpy array.
    """
    predictions, labels = p  # predictions: ndarray; labels: ndarray (with -100 indicating ignore)
    predictions_bin = np.argmax(predictions, axis=2)
    # Compute probabilities with softmax over the last axis
    predictions_prob_pos = softmax(predictions, axis=2)[:, :, 1]

    # Filter out ignored tokens (-100)
    true_predictions_bin = [
        [pred for (pred, lab) in zip(pred_seq, lab_seq) if lab != -100]
        for pred_seq, lab_seq in zip(predictions_bin, labels)
    ]
    true_labels = [
        [lab for lab in lab_seq if lab != -100] for lab_seq in labels
    ]
    true_predictions_prob_pos = [
        [prob for (prob, lab) in zip(prob_seq, lab_seq) if lab != -100]
        for prob_seq, lab_seq in zip(predictions_prob_pos, labels)
    ]

    # Compute the disorder metric for each sequence
    disorder = []
    for l, p in tqdm(zip(true_labels, true_predictions_bin), total=len(true_labels), desc="Computing disorder"):
        disorder.append(get_observed_disorder(l, p))
    # print('Number of NaN disorder:', np.sum(np.isnan(disorder)))

    # Compute standard metrics
    f1 = [f1_score(l, p, pos_label=1, average='binary') for l, p in zip(true_labels, true_predictions_bin)]
    precision = [precision_score(l, p, pos_label=1, average='binary') for l, p in zip(true_labels, true_predictions_bin)]
    recall = [recall_score(l, p, pos_label=1, average='binary') for l, p in zip(true_labels, true_predictions_bin)]
    accuracy = [accuracy_score(l, p) for l, p in zip(true_labels, true_predictions_bin)]
    auprc = [get_auprc(l, p) for l, p in zip(true_labels, true_predictions_prob_pos)]
    r_precision = [get_r_precision(p, l) for l, p in zip(true_labels, true_predictions_prob_pos)]
    # You can also add additional metrics (e.g., correlation) if needed

    return {
        "f1": np.nanmean(f1),
        "precision": np.nanmean(precision),
        "recall": np.nanmean(recall),
        "accuracy": np.nanmean(accuracy),
        "auprc": np.nanmean(auprc),
        "disorder": np.nanmean(disorder),
        "r_precision": np.nanmean(r_precision),
        "num_nan_disorder": np.sum(np.isnan(disorder)),
    }


def set_global_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AggTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute the loss for the model.
        """
        current_epoch = self.state.epoch if self.state is not None else 0
        inputs["epoch"] = current_epoch

        model_inputs = {k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask", "labels"]}
        return super().compute_loss(model, model_inputs, return_outputs=return_outputs, **kwargs)


def train_highlighter(
    args,
    model,
    retriever,
    tokenizer,
    train_agg_types: List[str] = ["naive"],
    validate_agg_type: str = "naive",
    compute_metrics_fn=compute_metrics,
    metric_for_best_model: str = "valid_f1",  # e.g., "valid_f1", "disorder", etc.
    greater_is_better: bool = True,
    append_to_training: bool= True,
    training_args_kwargs: Optional[dict] = None,
    tokenize_and_align_labels_fn=tokenize_and_align_labels,
    train_model: bool = True,
    resume_from_checkpoint: Optional[str] = None,
    seed: int = 42,
    data_collator: Optional[any] = None,
    inference_module: Optional[object] = None,
    device: torch.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu"),
    all_doc_embeddings: Optional[torch.Tensor] = None,
    top_k: int = 1,
    cik_to_name: Optional[dict] = None,
    final_texts: Optional[List[str]] = None,
):
    
    model.train()
    retriever.eval()
    
    """
    Train (or run inference) using the HuggingFace Trainer framework.

    Args:
        model: HuggingFace model instance.
        tokenizer: HuggingFace tokenizer instance.
        train_agg_types: List of aggregation types (e.g., ["naive", "loose"]) to be used as training labels.
        compute_metrics_fn: Function to compute evaluation metrics.
        metric_for_best_model: Metric used for best model selection.
        greater_is_better: Whether a higher metric value is better.
        training_args_kwargs: Dictionary for overriding default TrainingArguments.
        train_model: If False, skip training and run inference only.
        seed: Random seed.
        data_collator: Custom data collator; if None, a default AggDataCollatorForTokenClassification is used.
        inference_module: Optional module for specialized inference (e.g., attn_highlighter).
    Returns:
        If training: the Trainer instance after training.
        If inference-only: inference outputs.
    """
    # Set the random seed for reproducibility
    set_global_seed(seed)

    # Load the datasets
    datasets = get_raw_datasets(
        train_agg_types=train_agg_types, 
        validate_agg_type=validate_agg_type,
        tokenizer=tokenizer,
    )

    # Add the "train_append", "valid_append", "test_append", and "expert_append" datasets
    datasets.update(get_appended_datasets(
        retriever=retriever, 
        tokenizer=tokenizer, 
        all_doc_embeddings=all_doc_embeddings, 
        doc_texts=final_texts,
        train_agg_types=train_agg_types,
        validate_agg_type=validate_agg_type,
        cik_to_name=cik_to_name,
        top_k=top_k,
    ))

    dataset_dict = DatasetDict(datasets)

    # Tokenize and align labels
    def tokenize_and_align_labels_wrapper(examples):
        return tokenize_and_align_labels_fn(examples, tokenizer=tokenizer)

    tokenized_datasets = dataset_dict.map(tokenize_and_align_labels_wrapper, batched=True)

    # Use a default data collator if none is provided
    if data_collator is None:
        data_collator = AggDataCollatorForTokenClassification(tokenizer=tokenizer)
        
    
    # Set up TrainingArguments with defaults (overridable via training_args_kwargs)
    training_args_kwargs = training_args_kwargs or {}
    training_args = TrainingArguments(
        output_dir=training_args_kwargs.get("output_dir", "checkpoints/highlighter"),
        run_name=training_args_kwargs.get("run_name", "highlighter_training"),
        learning_rate=training_args_kwargs.get("learning_rate", 2e-5),
        per_device_train_batch_size=training_args_kwargs.get("train_batch_size", 16),
        per_device_eval_batch_size=training_args_kwargs.get("eval_batch_size", 16),
        num_train_epochs=training_args_kwargs.get("num_train_epochs", 1),
        weight_decay=training_args_kwargs.get("weight_decay", 0.01),
        label_names=["labels"],
        eval_strategy=training_args_kwargs.get("eval_strategy", "epoch"),
        save_strategy=training_args_kwargs.get("save_strategy", "epoch"),
        load_best_model_at_end=True,
        remove_unused_columns=False,
        report_to=training_args_kwargs.get("report_to", "wandb"),
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
    )

    # Inference-only mode: skip training and run inference on sample texts.
    '''
    if not train_model:
        sample_texts = ["This is a sample input text for highlighting inference."]
        if inference_module is not None:
            # Expecting the module to have a method `highlighting_outputs`
            return inference_module.highlighting_outputs(sample_texts)
        elif hasattr(model, "encode"):
            return model.encode(sample_texts)
        else:
            # Fallback: use a token-classification pipeline.
            pipe = pipeline("token-classification", model=model, tokenizer=tokenizer)
            return pipe(sample_texts)
    '''

    if append_to_training:
        training_data= tokenized_datasets["train_append"]
    elif not append_to_training:
        training_data= tokenized_datasets["train"]

    # Initialize the Trainer
    trainer = AggTrainer(
        model=model,
        args=training_args,
        train_dataset=training_data,
        eval_dataset={
            "train": tokenized_datasets["train"],
            #"valid": tokenized_datasets["valid"],
            #"test": tokenized_datasets["test"],
            #"expert": tokenized_datasets["expert"],
            #"train_append": tokenized_datasets["train_append"],
            #"valid_append": tokenized_datasets["valid_append"],
            #"test_append": tokenized_datasets["test_append"],
            #"expert_append": tokenized_datasets["expert_append"],
        },
        #data_collator=data_collator,
        data_collator=AggDataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics_fn,
        #callbacks=[
        #    EarlyStoppingCallback(
        #        early_stopping_patience=training_args_kwargs.get("early_stopping_patience", 5)
        #    )
        #],
    )

    # Train and then evaluate the model
    if train_model:
        trainer.train(ignore_keys_for_eval=["attentions", "hidden_states"], resume_from_checkpoint=resume_from_checkpoint)
    trainer.evaluate(ignore_keys=["attentions", "hidden_states"])
    trainer.save_model(training_args_kwargs.get("output_dir", "checkpoints/highlighter"))
    
    return trainer


# ===== Example usage =====
if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # Load your default model and tokenizer (or plug in any alternative)
    args = argparse.ArgumentParser()
    args.add_argument('--output_dir', '-o', type=str, default='checkpoints/highlighter')
    args.add_argument('--run_name', '-r', type=str, default='highlighter_training')
    args.add_argument('--learning_rate', '-lr', type=float, default=2e-5)
    args.add_argument('--num_train_epochs', '-n', type=int, default=50)
    args.add_argument('--early_stopping_patience', '-p', type=int, default=5)
    args.add_argument('--train_batch_size', '-b', type=int, default=16)
    args.add_argument('--eval_batch_size', '-e', type=int, default=16)
    args.add_argument('--weight_decay', '-w', type=float, default=0.01)
    args.add_argument('--eval_strategy', '-es', type=str, default='epoch')
    args.add_argument('--save_strategy', '-ss', type=str, default='epoch')
    args.add_argument('--report_to', '-rt', type=str, default='wandb')
    args.add_argument('--model_name', '-m', type=str, default='bert-base-uncased')
    args.add_argument('--train_model', '-tm', type=str2bool, default=True)
    args.add_argument('--seed', '-s', type=int, default=42)
    args.add_argument('--train_agg_types', '-t', nargs='+', default=["naive"])
    args.add_argument('--validate_agg_type', '-v', type=str, default='naive')
    args.add_argument('--metric_for_best_model', '-mbm', type=str, default='valid_f1')
    args.add_argument('--greater_is_better', '-gib', type=str2bool, default=True)
    args.add_argument('--agg_strategy', '-as', type=str, default='mix')
    args.add_argument('--agg_type_order', '-ato', nargs='+', default=['strict', 'loose'])
    args.add_argument('--agg_type_weights', '-atw', default=[0.5, 0.5])
    args.add_argument('--resume_from_checkpoint', '-rfc', type=str2bool, default=False)


    args = args.parse_args()

    
    import os
    os.environ["WANDB_PROJECT"]="fin.highlight"
    import wandb
    wandb.login()

    if args.resume_from_checkpoint:
        # Load the model from the checkpoint
        if args.model_name == 'agg_highlighter':
            model = AggHighlighter.from_pretrained(args.output_dir)
        else:
            try:
                model = AutoModelForTokenClassification.from_pretrained(args.output_dir)
            except Exception as e:
                base_model = BertForTokenClassification.from_pretrained(args.output_dir)
                model = BertForTokenClassificationWrapper(base_model) 

        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        tokenize_and_align_fn = tokenize_and_align_labels
        data_collator = None

    elif args.model_name == 'bert-base-uncased':
        from transformers import BertForTokenClassification
        base_model = BertForTokenClassification.from_pretrained(
            args.model_name,
            num_labels=len(ID2LABEL),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        model = BertForTokenClassificationWrapper(base_model)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        data_collator = None
        tokenize_and_align_fn = tokenize_and_align_labels

    elif args.model_name == 'agg_highlighter':
        config = AutoConfig.from_pretrained('bert-base-uncased')
        model = AggHighlighter(
            config,
            agg_weights_base=args.agg_type_weights,
            type_order=args.agg_type_order,
            strategy=args.agg_strategy,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            num_labels=len(ID2LABEL),
        )
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        data_collator = None
        tokenize_and_align_fn = tokenize_and_align_labels


    elif args.model_name == 'cnc_highlighter':
        # model = BertForHighlightPrediction.from_pretrained('DylanJHJ/bert-base-final-v0-ep2')
        model = CncHighlighterWrapper('DylanJHJ/bert-base-final-v0-ep2')
        tokenizer = AutoTokenizer.from_pretrained('DylanJHJ/bert-base-final-v0-ep2')
        # data_collator = CncDataCollatorForTokenClassification(tokenizer=tokenizer)
        data_collator = None
        tokenize_and_align_fn = tokenize_and_align_labels_cnc

    print("Training or not training:", args.train_model)
    print("Model name:", args.model_name)


    trainer_obj = train_highlighter(
        model=model,
        tokenizer=tokenizer,
        train_agg_types=args.train_agg_types,
        validate_agg_type=args.validate_agg_type,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        tokenize_and_align_labels_fn=tokenize_and_align_fn,
        training_args_kwargs={
            "output_dir": args.output_dir,
            "run_name": args.run_name,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "weight_decay": args.weight_decay,
            "eval_strategy": args.eval_strategy,
            "save_strategy": args.save_strategy,
            "report_to": args.report_to,
            "resume_from_checkpoint": args.resume_from_checkpoint,
        },
        train_model=args.train_model,
        data_collator=data_collator,
        seed=args.seed,
    )



    # Optionally, you can prepare an inference module (e.g., an attn_highlighter instance)
    inference_module = None  # Replace with your module instance if needed

    # If training, trainer_obj is the Trainer instance.
    print("Trainer running complete.")

