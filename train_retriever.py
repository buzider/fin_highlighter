import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BertForTokenClassification
import random
import os
from tqdm import tqdm
import json
import argparse
import numpy as np
import pickle
import logging
import warnings
import statistics
import time
import copy

from evaluation.metrics import (
    get_observed_disorder, 
    get_auprc, 
    get_r_precision, 
    get_correlation
)
from def_functions import (
    embed_texts_contriever,
    test_memory_allocation,
    embed_texts_contriever2,
    mean_pooling,
    InfoNCELoss,
)


def generate_contrastive_data(
    training_annotated_results,
    final_texts,
    cik_to_name,
    model,
    retriever,
    tokenizer,
    device,
    text_size=1000,
    topk=5,
    num_negatives=80,
    batch_size=16,
    criterion_hlt=None,
    criterion_ret=None,
):
    import statistics
    contrastive_learning_data = []
    contrastive_learning_valid_data = []

    model.eval()
    
    for idx, training_element in enumerate(training_annotated_results):

        shuffled_texts = final_texts.copy()
        random.shuffle(shuffled_texts)
        selected_texts = shuffled_texts[:text_size]

        query_firm_name = cik_to_name[training_element['id'].split('_')[2]]
        concat_text = f"{query_firm_name} {training_element['texts']}"

        tokenized_ids = tokenizer.convert_tokens_to_ids(training_element['tokens'])
        tokenized_stringA = torch.tensor(tokenized_ids).to(device)
        if tokenized_stringA.size(0) > 250:
            tokenized_stringA = tokenized_stringA[:250]
        seq_len_A = tokenized_stringA.size(0)

        true_label_tensor = torch.tensor(training_element['labels']).float().to(device)
        if true_label_tensor.size(0) > 250:
            true_label_tensor = true_label_tensor[:250]
        valid_mask = true_label_tensor != -100

        sep_token_id = tokenizer.sep_token_id
        sep_token_tensor = torch.tensor([sep_token_id], device=device)

        top_max_texts = []

        with torch.no_grad():
            for i in range(0, len(selected_texts), batch_size):
                batch_texts = selected_texts[i:i + batch_size]
                batch_tokenized = tokenizer(
                    batch_texts,
                    add_special_tokens=False,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=250
                )
                tokenized_stringB = batch_tokenized['input_ids'].to(device)

                combined_input_ids = [
                    torch.cat([tokenized_stringA, sep_token_tensor, b]) for b in tokenized_stringB
                ]
                combined_input_ids_padded = torch.nn.utils.rnn.pad_sequence(
                    combined_input_ids,
                    batch_first=True,
                    padding_value=tokenizer.pad_token_id
                ).to(device)

                attention_mask = (combined_input_ids_padded != tokenizer.pad_token_id).long().to(device)

                outputs = model(
                    input_ids=combined_input_ids_padded,
                    attention_mask=attention_mask
                )
                logits = outputs.logits

                logits_before_sep = logits[:, :seq_len_A, :]
                probs = torch.sigmoid(logits_before_sep)[..., 1]

                for j in range(len(batch_texts)):
                    probs_j = probs[j]
                    valid_mask_cut = valid_mask[:probs_j.size(0)]
                    true_label_tensor_cut = true_label_tensor[:probs_j.size(0)]
                    loss = criterion_hlt(probs_j[valid_mask_cut], true_label_tensor_cut[valid_mask_cut])
                    top_max_texts.append((loss.item(), batch_texts[j]))

        top_max_texts_sorted = sorted(top_max_texts, key=lambda x: x[0])
        
        # logging
        values = [item[0] for item in top_max_texts_sorted]
        #print(f"Loss min/median/max: {values[0]:.4f} / {statistics.median(values):.4f} / {values[-1]:.4f}")

        for i in range(topk):
            tmp = [
                training_element['texts'],
                concat_text,
                top_max_texts_sorted[i][1],
                random.sample(shuffled_texts, num_negatives)
            ]
            contrastive_learning_data.append(tmp)

        tmp_val = [
            training_element['texts'],
            concat_text,
            top_max_texts_sorted[topk][1],
            random.sample(shuffled_texts, num_negatives)
        ]
        contrastive_learning_valid_data.append(tmp_val)

    return contrastive_learning_data, contrastive_learning_valid_data





def train_retriever(
    model,
    retriever,
    tokenizer,
    training_annotated_results: list,
    final_texts: list[str],
    cik_to_name: dict,
    device,
    #optimizer,
    #criterion,
    text_size: int = 1000,
    batch_size: int = 10,
    epochs: int = 1,
    patience: int = 5,
    args: any=None,
    time_records=None, 
    log_time_fn=None,
    checkpoint=0,
):
    contrastive_learning_data = []
    contrastive_learning_valid_data = []
    count = 0

    optimizer = torch.optim.Adam(
    list(retriever.parameters()), lr=args.lr
)   
    criterion_hlt = torch.nn.BCELoss()
    criterion_ret = InfoNCELoss(temperature=0.05).to(device)
    
    model.eval()
    if log_time_fn:
        log_time_fn(f'Round {checkpoint+1}, Start get contrastive learning data! ')
    
    contrastive_learning_data, contrastive_learning_valid_data = generate_contrastive_data(
        training_annotated_results,
        final_texts,
        cik_to_name,
        model,
        retriever,
        tokenizer,
        device,
        text_size=text_size,
        topk=5,
        num_negatives=80,
        batch_size=batch_size,
        criterion_hlt=criterion_hlt,
        criterion_ret=criterion_ret
    )
    if log_time_fn:
        log_time_fn(f'Round {checkpoint+1}, Finish get contrastive learning data!')
    # early stopping 相關變數
    best_val_loss = float('inf')
    counter = 0
    best_retriever_state = copy.deepcopy(retriever.state_dict())

    for epoch in range(epochs):
        random.shuffle(contrastive_learning_data)
        random.shuffle(contrastive_learning_valid_data)
        model.eval()
        retriever.train()
        for i, item in tqdm(enumerate(contrastive_learning_data), total=len(contrastive_learning_data), desc=f"contrastive learning"):
            while True:
                if test_memory_allocation(torch.device(device), size_mb=2048):
                    reserved_memory = torch.empty((int(2 * 1024**3 / 4),), dtype=torch.float32, device=device)  # 預留 2GB
                    #print(f"GPU {args.cuda} has enough memory.")
                    break
                else:
                    #print("GPU memory is not enough，wait for 10 秒")
                    time.sleep(10)
                    continue

            #print(f'Contrastive: Epoch {epoch + 1}, Step {i + 1}')
            original, anchor, positive, negatives = item  # Unpack the item

            anchor_embedding = embed_texts_contriever(anchor, retriever, tokenizer, device)
            positive_embedding = embed_texts_contriever(positive, retriever, tokenizer, device)

            # Assuming negatives is a list of negative samples
            with torch.no_grad():
                try:
                    negative_embeddings = embed_texts_contriever(negatives, retriever, tokenizer, device)  # Shape: (30, 768)
                    negative_embeddings = negative_embeddings.unsqueeze(0)  # Correct shape: (1, 30, 768)
                    del reserved_memory
                    torch.cuda.empty_cache()
                except RuntimeError:
                    del reserved_memory
                    torch.cuda.empty_cache()
                    negative_embeddings = embed_texts_contriever(negatives, retriever, tokenizer, device)  # Shape: (30, 768)
                    negative_embeddings = negative_embeddings.unsqueeze(0)  # Correct shape: (1, 30, 768)
                    

            loss = criterion_ret(anchor_embedding, positive_embedding, negative_embeddings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(f'Contrastive: Epoch {epoch + 1}, Step {i + 1} Done, Loss = {loss.item()}')
        
        if log_time_fn:
            log_time_fn(f'Round {checkpoint+1}, retriever training, Contrastive: Done, Loss = {loss.item()}')

        # 驗證 retriever（optional）
        # 可以以 validation loss、或預先選定 anchor/positive 進行簡易評估
        validation_loss = 0.0
        for item in tqdm(contrastive_learning_valid_data, desc=f"contrastive learning validation", total=len(contrastive_learning_valid_data)):
            while True:
                if test_memory_allocation(torch.device(device), size_mb=2048):
                    reserved_memory = torch.empty((int(2 * 1024**3 / 4),), dtype=torch.float32, device=device)  # 預留 2GB
                    #print(f"GPU {args.cuda} has enough memory.")
                    break
                else:
                    #print("GPU memory is not enough，wait for 10 秒")
                    time.sleep(10)
                    continue
            original, anchor, positive, negatives = item            
            anchor_embedding = embed_texts_contriever(anchor, retriever, tokenizer, device)
            positive_embedding = embed_texts_contriever(positive, retriever, tokenizer, device)
            with torch.no_grad():
                try:
                    negative_embeddings = embed_texts_contriever(negatives, retriever, tokenizer, device).unsqueeze(0)
                    del reserved_memory
                    torch.cuda.empty_cache()
                except RuntimeError:
                    del reserved_memory
                    torch.cuda.empty_cache()
                    negative_embeddings = embed_texts_contriever(negatives, retriever, tokenizer, device)  # Shape: (30, 768)
                    negative_embeddings = negative_embeddings.unsqueeze(0)  # Correct shape: (1, 30, 768)
                            
            
            loss = criterion_ret(anchor_embedding, positive_embedding, negative_embeddings)
            validation_loss += loss.item()

        avg_val_loss = validation_loss / len(contrastive_learning_valid_data)
        #print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}")

        if log_time_fn:
            log_time_fn(f"Round {checkpoint+1}, retriever training, Validation Loss: {avg_val_loss:.4f}")

        '''#Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            best_retriever_state = copy.deepcopy(retriever.state_dict())
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    retriever.load_state_dict(best_retriever_state)
    '''
    retriever.eval()
    model.eval()
        
    batch_size = 512  # Define the desired batch size
    all_doc_embeddings = []
    
    if log_time_fn:
        log_time_fn(f'Round {checkpoint+1}, Start compute embeddings!')
    # Iterate over final_texts in chunks of batch_size
    for i in tqdm(range(0, len(final_texts), batch_size), desc="Computing embeddings"):
        #print(f'{i}/{len(final_texts)}')
        batch = final_texts[i:i + batch_size]
        batch_embeddings = embed_texts_contriever2(batch, retriever, tokenizer, device)
        all_doc_embeddings.append(batch_embeddings) 
    all_doc_embeddings = torch.cat(all_doc_embeddings, dim=0)
    all_doc_embeddings = F.normalize(all_doc_embeddings, p=2, dim=1) 

    if log_time_fn:
        log_time_fn(f'Round {checkpoint+1}, Finish compute embeddings!')

    #print(f'ret training ended on device: {device}')
    return retriever, all_doc_embeddings
