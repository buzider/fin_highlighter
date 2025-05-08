import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from evaluation.metrics import (
    get_observed_disorder,
    get_auprc,
    get_r_precision,
    get_correlation
)
from def_functions import embed_texts_contriever2

def test(
    model,
    retriever,
    tokenizer,
    device,
    annotated_results,
    all_doc_embeddings,
    cik_to_name,
    final_texts,
    args,
    text_size,
    seed,
    rank,
    dataset_name,
    retriever_training=False,
    checkpoint=0,
    epoch_h=0,
    epoch_r=0,
    time_records=None,
    log_time_fn=None,
):
    precision_list, recall_list, f1_list = [], [], []
    disorder_list, auprc_list, r_precision_list, corr_list = [], [], [], []

    model.eval()
    if log_time_fn:
        model_type = "retriever" if retriever_training else "highlighter"
        log_time_fn(f"{checkpoint+1}_{model_type} rank{rank} {dataset_name} test start!")

    count = 0
    for element in annotated_results:
        count += 1
        #print(f'testing: {count}')

        tokenized_ids = tokenizer.convert_tokens_to_ids(element['tokens'])[:512]
        tokenized_input = torch.tensor(tokenized_ids).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(tokenized_input).to(device)

        with torch.no_grad():
            outputs = model(input_ids=tokenized_input, attention_mask=attention_mask)
            logits = outputs.logits  # (1, seq_len, 2)
            probs = torch.sigmoid(logits)[0, :, 1]

            true_label_tensor = torch.tensor(element['labels']).float().to(device)

            min_len = min(len(probs), len(true_label_tensor))
            probs = probs[:min_len]
            true_label_tensor = true_label_tensor[:min_len]

            valid_mask = true_label_tensor != -100
            filtered_probs = probs[valid_mask]
            filtered_labels = true_label_tensor[valid_mask]
            binary_tensor = (filtered_probs >= 0.5).float()


            # Custom metrics
            disorder = get_observed_disorder(filtered_labels, binary_tensor)
            auprc = get_auprc(filtered_labels.cpu().numpy(), filtered_probs.cpu().numpy())
            r_precision = get_r_precision(filtered_labels.cpu().numpy(), filtered_probs.cpu().numpy())
            correlation = get_correlation(filtered_labels.cpu().numpy(), filtered_probs.cpu().numpy())
            
            # Sklearn metrics
            precision = precision_score(filtered_labels.cpu(), binary_tensor.cpu(), zero_division=0)
            recall = recall_score(filtered_labels.cpu(), binary_tensor.cpu(), zero_division=0)
            f1 = f1_score(filtered_labels.cpu(), binary_tensor.cpu(), zero_division=0)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        disorder_list.append(disorder)
        auprc_list.append(auprc)
        r_precision_list.append(r_precision)
        corr_list.append(correlation)

        #print(f'finished:checkpoint_H_{checkpoint+1}_{epoch_h}_R_{checkpoint+int(retriever_training)}_{epoch_r}_textsize_{text_size}_{dataset_name}_result_top_{rank}_seed_{seed}_{count}')

    result = {
        "count": count,
        "precision": np.nanmean(precision_list),
        "recall": np.nanmean(recall_list),
        "f1": np.nanmean(f1_list),
        "disorder": np.nanmean(disorder_list),
        "auprc": np.nanmean(auprc_list),
        "r_precision": np.nanmean(r_precision_list),
        "corr": np.nanmean(corr_list),
        "num_nan_disorder": np.sum(np.isnan(disorder_list)),
        "precision_list": precision_list,
        "recall_list": recall_list,
        "f1_list": f1_list,
        "disorder_list": disorder_list,
        "auprc_list": auprc_list,
        "r_precision_list": r_precision_list,
        "corr_list": corr_list
    }

    output_dir = os.path.join(f"/home/teyu/Fin_highlighter_new/rag_result_iterative/{args.testnum}/{seed}", dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"checkpoint_{checkpoint+1}_{epoch_h}_{checkpoint+int(retriever_training)}_{epoch_r}_textsize_{text_size}_{dataset_name}_result_top_{rank}_seed_{seed}.json"
    output_path = os.path.join(output_dir, output_file)

    with open(output_path, "w", encoding="utf-8") as json_file:
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        json.dump(result, json_file, ensure_ascii=False, indent=4, default=convert_numpy)
        #print(f'data storing finished at: {output_path}')
    if log_time_fn:
        model_type = "retriever" if retriever_training else "highlighter"
        log_time_fn(f"{checkpoint+1}_{model_type} rank{rank} {dataset_name} dataset test finished!")
    if dataset_name == 'validation':
        return np.nanmean(f1_list) if args.early_stopping_metrics == 'f1' else np.nanmean(disorder_list)
    else:
        return 0 if args.early_stopping_metrics== 'f1' else 1

