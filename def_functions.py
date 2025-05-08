
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BertForTokenClassification
import random
import os
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



def test_memory_allocation(device, size_mb=500):
    torch.cuda.empty_cache()  # 先清理快取
    torch.cuda.synchronize()  # 等待清理完成
    try:
        tensor = torch.empty((size_mb * 1024 * 1024 // 4,), dtype=torch.float32, device=device)
        del tensor  # 釋放內存
        torch.cuda.empty_cache()  # 先清理快取
        torch.cuda.synchronize()  # 等待清理完成
        return True
    except RuntimeError:
        return False


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def embed_texts_contriever(text, model, tokenizer, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    emb = mean_pooling(model(**inputs)[0], inputs['attention_mask'])
    
    return emb

def embed_texts_contriever2(text, model, tokenizer, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        emb = mean_pooling(model(**inputs)[0], inputs['attention_mask'])
    
    return emb

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, anchor, positive, negatives):
        """
        Args:
            anchor: Tensor of shape (batch_size, embedding_dim)
            positive: Tensor of shape (batch_size, embedding_dim)
            negatives: Tensor of shape (batch_size, num_negatives, embedding_dim)
        
        Returns:
            loss: InfoNCE loss value
        """
        batch_size = anchor.size(0)
        
        # Normalize embeddings
        anchor_norm = F.normalize(anchor, dim=1)
        positive_norm = F.normalize(positive, dim=1)
        negatives_norm = F.normalize(negatives, dim=2)
        
        # Compute positive logits: (batch_size, 1)
        positive_logits = torch.sum(anchor_norm * positive_norm, dim=1, keepdim=True) / self.temperature
        
        # Compute negative logits: (batch_size, num_negatives)
        # For each anchor, compute similarity with all negatives
        anchor_expanded = anchor_norm.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        negative_logits = torch.bmm(anchor_expanded, negatives_norm.transpose(1, 2)).squeeze(1) / self.temperature  # (batch_size, num_negatives)
        
        # Concatenate positive and negative logits: (batch_size, 1 + num_negatives)
        logits = torch.cat([positive_logits, negative_logits], dim=1)
        
        # Labels: positives are the first element in each row
        labels = torch.zeros(batch_size, dtype=torch.long).to(anchor.device)
        
        # Apply cross-entropy loss
        loss = self.cross_entropy(logits, labels)
        
        return loss
    
def store_model(args, model, checkpoint, epoch_count=1):

    save_dir = f"/home/teyu/Fin_highlighter_new/rag_model/{args.testnum}/{args.seed}/highlighter"
    os.makedirs(save_dir, exist_ok=True)  

    # store Highlighter
    torch.save(model.state_dict(), os.path.join(save_dir, f"model_checkpoint_{checkpoint+1}_{epoch_count}.pth"))


def store_retriever(args, retriever, checkpoint, epoch_count=1):
    
    save_dir = f"/home/teyu/Fin_highlighter_new/rag_model/{args.testnum}/{args.seed}/retriever"
    os.makedirs(save_dir, exist_ok=True)  

    # store Retriever
    torch.save(retriever.state_dict(), os.path.join(save_dir, f"retriever_checkpoint_{checkpoint+1}_{epoch_count}.pth"))

def store_best_model(args,model):

    save_dir = f"/home/teyu/Fin_highlighter_new/rag_model/{args.testnum}/{args.seed}/highlighter"
    os.makedirs(save_dir, exist_ok=True)
    # store Highlighter
    torch.save(model.state_dict(), os.path.join(save_dir, f"best_model.pth"))

def store_best_retriever(args,retriever):
    save_dir = f"/home/teyu/Fin_highlighter_new/rag_model/{args.testnum}/{args.seed}/retriever"
    os.makedirs(save_dir, exist_ok=True)
    # store Retriever
    torch.save(retriever.state_dict(), os.path.join(save_dir, f"best_retriever.pth"))
