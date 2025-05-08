import json
import random
import pandas as pd
from pathlib import Path
from pprint import pprint
from typing import List
from collections import Counter
import torch
import torch.nn.functional as F
from transformers import DataCollatorForTokenClassification
from transformers.data.data_collator import DataCollatorMixin, pad_without_fast_tokenizer_warning
from transformers import BertTokenizerFast
from datasets import Dataset, Features, Value, Sequence

from highlighter.cnc_full_highlighter import CncAlignment
from utils.utils import read_jsonl, retrieve_paragraph_from_docid
DATA_DIR = Path('annotation/annotated_result/all/setting2/')
TRAIN_DATA = DATA_DIR / 'train.jsonl'
VALID_DATA = DATA_DIR / 'valid.jsonl'
TEST_DATA = DATA_DIR / 'test.jsonl'
EXPERT_DATA = DATA_DIR / 'expert.jsonl'
AGG_MAP = {
    'strict': 0,
    'complex': 1,
    'harsh': 2,
    'naive': 3,
    'loose': 4,
    # 'expert': 5
}
LABEL_LIST = ['0', '1']
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
LABEL2ID = {label: i for i, label in ID2LABEL.items()}

def illustrate_a_sample(tokenized_datasets, model, data_collator, index=0):
    import torch.nn.functional as F
    # prepare case study samples
    model.eval()
    model.to('cpu')
    train_sample = data_collator.torch_call([tokenized_datasets['train'][index]])
    train_sample_label = train_sample['labels'].tolist()
    valid_sample = data_collator.torch_call([tokenized_datasets['valid'][index]])
    valid_sample_label = valid_sample['labels'].tolist()
    expert_sample = data_collator.torch_call([tokenized_datasets['test'][index]])
    expert_sample_label = expert_sample['labels'].tolist()

    train_pred = F.sigmoid(model(**train_sample).logits[0]).max(1).indices.tolist()
    valid_pred = F.sigmoid(model(**valid_sample).logits[0]).max(1).indices.tolist()
    expert_pred = F.sigmoid(model(**expert_sample).logits[0]).max(1).indices.tolist()
    print('Train Sample:')
    print(f'Label: {train_sample_label}')
    print(f'Prediction: {train_pred}')
    print('Valid Sample:')
    print(f'Label: {valid_sample_label}')
    print(f'Prediction: {valid_pred}')
    print('Expert Sample:')
    print(f'Label: {expert_sample_label}')
    print(f'Prediction: {expert_pred}')


# load the data
def read_setting2_data():
    train_data = read_jsonl(TRAIN_DATA)
    valid_data = read_jsonl(VALID_DATA)
    test_data = read_jsonl(TEST_DATA)
    expert_data = read_jsonl(EXPERT_DATA)
    return train_data, valid_data, test_data, expert_data


def data_generator_mix_all(data_list, aggregation_labels=['strict_aggregation', 'complex_aggregation', 'harsh_aggregation', 'naive_aggregation', 'loose_aggregation']):
    for data_idx, data in enumerate(data_list):
        has_valid_label = False
        for key in data:
            if 'aggregation' in key:
                if key not in aggregation_labels:
                    continue
                labels = data[key]['label']
                has_valid_label = True
                yield {
                    'id': data['sample_id'],
                    'texts': data['text'],
                    'tokens': data['tokens'],
                    'labels': labels,
                    'highlight_probs': data['highlight_probs'],
                    'aggregation': AGG_MAP[key.split('_')[0]],  # 🚨 這裡直接轉成 int！
                }
        if not has_valid_label:
            print(f"[WARNING] No valid aggregation label found for {data.get('sample_id')}")

def data_generator_mix_all_debug(data_list, aggregation_labels=None):

    if aggregation_labels is None:
        aggregation_labels = [
            'strict_aggregation',
            'complex_aggregation',
            'harsh_aggregation',
            'naive_aggregation',
            'loose_aggregation'
        ]

    print(f"🧪 接收到資料筆數：{len(data_list)}")

    total_yielded = 0

    for i, data in enumerate(data_list):
        tokens = data.get('tokens')
        id_ = data.get('sample_id')
        text = data.get('text')

        matched = False

        for key in data:
            if 'aggregation' in key:
                print(f"🔍 第{i}筆找到 aggregation key：{key}")

                if key not in aggregation_labels:
                    print(f"⚠️  key `{key}` 不在 aggregation_labels 中，跳過")
                    continue

                entry = data.get(key)
                if not isinstance(entry, dict):
                    print(f"⚠️  key `{key}` 的內容不是 dict，跳過")
                    continue

                labels = entry.get('label')
                if not isinstance(labels, list):
                    print(f"⚠️  key `{key}` 的 label 欄不是 list，跳過")
                    continue

                if all(l is None for l in labels):
                    print(f"⚠️  key `{key}` 的所有 label 都是 None，跳過")
                    continue

                print(f"✅ yield 第{i}筆的 `{key}`")
                total_yielded += 1

                yield {
                    'id': id_,
                    'tokens': tokens,
                    'labels': labels,
                    'aggregation': key,
                    'texts': text
                }
                matched = True

        if not matched:
            print(f"❌ 第{i}筆資料完全沒有符合條件的 aggregation key")
            pass
    print(f"📦 generator 結束，總共 yield {total_yielded} 筆資料")



def data_generator_expert(data_list):
    for data in data_list:
        id_ = list(data.keys())[0]
        tokens = data[id_]['tokens']
        texts = data[id_]['text']
        labels = data[id_]['binary_labels']
        yield {'id': id_, 'tokens': tokens, 'texts': texts, 'labels': labels, 'aggregation': 'expert'}


def tokenize_and_align_labels_cnc(examples, tokenizer, topK=1):
    '''
    examples is dictionary of list(len=batch_size)
    e.g. 'aggregation': ['expert', 'expert', 'expert', ...]
    '''
    cnc_alignment = CncAlignment(topK=topK)
    
    references = cnc_alignment.align_a_batch(examples) # list of (doc_id, rouge_score) # this assume only one input target, but in fact, there are multiple targets in examples # align is for one sample
    if topK > 1:
        raise NotImplementedError('topK > 1 is not implemented yet')
    else:
        # NOTE: for now, only use the first one
        references = [r[0] for r in references]
    references_texts = [retrieve_paragraph_from_docid(doc_id) for doc_id, _ in references]
    references_words = [r.split() for r in references_texts]
    target_words = examples['tokens']
    references_size = len(references_words)
    assert references_size == len(target_words), f"references size {references_size} != target size {len(target_words)}"
    tokenized_inputs = tokenizer(
        references_words, target_words,
        is_split_into_words=True,
        truncation=True,
        max_length=512, # for BERT
        # padding=True if tokenizer.padding_side == 'right' else False,
    )
    # print(tokenized_inputs.keys())
    # print(tokenizer.decode(tokenized_inputs['input_ids'][0]))
    # NOTE: implement for top1 for now
    # NOTE: not sure if padding is needed here
    labels = [] # labels for this batch
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        segment_ids = tokenized_inputs['token_type_ids'][i]
        # print(word_ids)
        # print(segment_ids)
        previous_word_idx = None
        label_ids = [] # labels for this sample
        for word_idx, seg_idx in zip(word_ids, segment_ids):
            if seg_idx == 0: # for the ref sentence
                label_ids.append(-100)
            elif seg_idx == 1: # for the target sentence
                if word_idx is None:
                    label_ids.append(-100)
                elif label[word_idx] is None:
                    label_ids.append(-100)

                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    # operate dictionary in this stage instead of in gpu
    tokenized_inputs['aggregation'] = [AGG_MAP[agg] if agg in AGG_MAP else 0 for agg in examples['aggregation']]

    return tokenized_inputs


def tokenize_and_align_labels(examples, tokenizer):
    

    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None or label[word_idx] is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    tokenized_inputs["id"] = examples["id"]
    tokenized_inputs["texts"] = examples["texts"]
    if "highlight_probs" in examples:
        tokenized_inputs["highlight_probs"] = examples["highlight_probs"]
    tokenized_inputs["aggregation"] = examples["aggregation"]

    
    return tokenized_inputs


# see: https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/data/data_collator.py#L288
# see: https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorForTokenClassification
# see: https://pytorch.org/docs/stable/data.html
# see: https://pytorch.org/docs/stable/_modules/torch/utils/data/_utils/collate.html#default_collate
# see: carefully check the relation between collate_fn and batch and model input, you can do it!
class AggDataCollatorForTokenClassification(DataCollatorForTokenClassification):
    def __init__(
        self, tokenizer, padding=True, max_length=None,
        pad_to_multiple_of=None, label_pad_token_id=-100, return_tensors="pt"
    ):
        super().__init__(
            tokenizer=tokenizer,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            label_pad_token_id=label_pad_token_id,
            return_tensors=return_tensors
        )

    def torch_call(self, features):
        import torch
        import torch.nn.functional as F

        label_name = "label" if "label" in features[0] else "labels"
        labels = [f[label_name] for f in features] if label_name in features[0] else None

        # Drop everything except input_ids + attention_mask
        no_label_features = [
            {k: f[k] for k in ["input_ids", "attention_mask"] if k in f}
            for f in features
        ]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            no_label_features,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )

        # Optional: drop unwanted keys
        for k in ["epoch", "aggregation"]:
            if k in batch:
                del batch[k]

        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(x):
            return x.tolist() if isinstance(x, torch.Tensor) else list(x)

        if padding_side == "right":
            batch[label_name] = [
                to_list(lab) + [self.label_pad_token_id] * (sequence_length - len(lab))
                for lab in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(lab)) + to_list(lab)
                for lab in labels
            ]

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)

        # Just to be safe: keep only required keys
        allowed_keys = ["input_ids", "attention_mask", label_name]
        batch = {k: v for k, v in batch.items() if k in allowed_keys}

        return batch

 

def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions



def get_statistics(data_path: str):
    '''
    file format: JSONL, e.g.
    {
        "id": "20221028_10-K_320193_part2_item7_para1",
        "text": "The Company is a leading global provider of technology solutions for the energy industry...",
        "tokens": ["The", "Company", "is", "a", "leading", ...],
        "highlight_probs": [0.0, 0.0, 1.0, 0.33, 0.0, ...],
        "highlight_labels": [0, 0, 1, 1, 0, ...],
        "highlight_spans", ["is a leading global", "energy industry", ...]
        "type": "0" # or "1", "2", "3", "4"
        "topic": "1" # or "2", "3", "4", "5", ....
        "subtopic": "7-2" ... # might be empty
    }
    '''
    statistics = {}

    # get data
    data = read_jsonl(data_path)

    # get type statistics
    type_count = Counter([item['type'] for item in data])
    ratio = {k: v/len(data) for k, v in type_count.items()}
    statistics['type'] = {
        'count': type_count,
        'ratio': ratio
    }

    # get topic statistics
    topic_count = Counter([item['topic'] for item in data])
    ratio = {k: v/len(data) for k, v in topic_count.items()}
    statistics['topic'] = {
        'count': topic_count,
        'ratio': ratio
    }

    # get subtopic statistics # need to count for None
    subtopic_count = Counter([item['subtopic'] for item in data if item['subtopic']])
    ratio = {k: v/len(data) for k, v in subtopic_count.items()}
    statistics['subtopic'] = {
        'count': subtopic_count,
        'ratio': ratio
    }

    # get length statistics, spans/all tokens
    length = [len(item['tokens']) for item in data]
    statistics['length'] = {
        'min': min(length),
        'max': max(length),
        'mean': sum(length) / len(data),
        'median': pd.Series(length).median()
    }

    # span length
    span_length = [len(span.split()) for item in data for span in item['highlight_spans']]
    statistics['span_length'] = {
        'min': min(span_length),
        'max': max(span_length),
        'mean': sum(span_length) / len(span_length),
        'median': pd.Series(span_length).median()
    }

    # number of spans
    num_span = [len(item['highlight_spans']) for item in data]

    # get the signal density statistics
    signal_density = [sum(item['highlight_probs']) / len(item['highlight_probs']) for item in data]
    statistics['signal_density'] = {
        'min': min(signal_density),
        'max': max(signal_density),
        'mean': sum(signal_density) / len(data),
        'median': pd.Series(signal_density).median()
    }

    # return dictionary
    return statistics


def slice_data(
        data_path: str, 
        ratio: List[float], 
        target_dir: str = None,
        seed: int = 666
    ):
    '''
    data_path: str
    ratio: ratio for training, validation, test, e.g. [0.8, 0.1, 0.1]
    target_dir: str
    seed: int
    '''
    data_path = Path(data_path)
    # set seed
    random.seed(seed)

    # get data
    data = read_jsonl(data_path)

    # shuffle data
    random.shuffle(data)

    # get number of each split
    total = len(data)
    train_num = int(total * ratio[0])
    test_num = int(total * ratio[2])
    valid_num = total - train_num - test_num

    # slice data
    data_splits = {
        'train': data[:train_num],
        'valid': data[train_num:train_num+valid_num],
        'test': data[train_num+valid_num:]
    }

    # assert total number of data
    assert len(data_splits['train']) + len(data_splits['valid']) + len(data_splits['test']) == total

    # save data
    if not target_dir:
        target_dir = f'{data_path.parent}/slice_seed{seed}'
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for i, split in enumerate(['train', 'valid', 'test']):
        split_data_path = target_dir / f'{split}.jsonl'
        with open(split_data_path, 'w') as f:
            for item in data_splits[i]:
                f.write(json.dumps(item) + '\n')

    return data_splits


def read_slice_data(data_dir: str):
    data_dir = Path(data_dir)
    data_splits = {}
    for split in ['train', 'valid', 'test']:
        split_data_path = data_dir / f'{split}.jsonl'
        data_splits[split] = read_jsonl(split_data_path)
    return data_splits



def get_raw_datasets(
    train_agg_types, 
    validate_agg_type, 
    cache_dir='./data_cache',
    tokenizer=None,
):
    def tokenize_and_align_labels_wrapper(examples):
        return tokenize_and_align_labels(examples, tokenizer=tokenizer)

    train_data, valid_data, test_data, expert_data = read_setting2_data()

    agg_labels = [f"{agg}_aggregation" for agg in train_agg_types]
    validate_agg_type = f"{validate_agg_type}_aggregation"

    features = Features({
        'id': Value('string'),
        'tokens': Sequence(Value('string')),
        'texts': Value('string'),
        'labels': Sequence(Value('int32')),
        'highlight_probs': Sequence(Value('float32')),
        'aggregation': Value('int64')  # 🚨 aggregation明確是int了！
    })

    train_dataset = Dataset.from_generator(
        data_generator_mix_all,
        gen_kwargs={'data_list': train_data, 'aggregation_labels': agg_labels},
        features=features,
        cache_dir=cache_dir
    )
    valid_dataset = Dataset.from_generator(
        data_generator_mix_all,
        gen_kwargs={'data_list': valid_data, 'aggregation_labels': [validate_agg_type]},
        features=features,
        cache_dir=cache_dir
    )
    test_dataset = Dataset.from_generator(
        data_generator_mix_all,
        gen_kwargs={'data_list': test_data, 'aggregation_labels': [validate_agg_type]},
        features=features,
        cache_dir=cache_dir
    )
    expert_dataset = Dataset.from_generator(
        data_generator_expert,
        gen_kwargs={'data_list': expert_data},
        cache_dir=cache_dir
    )

    datasets = {
        "train": train_dataset,
        "valid": valid_dataset,
        "test": test_dataset,
        "expert": expert_dataset,
    }

    for key in ["train", "valid", "test"]:
        datasets[key] = datasets[key].map(tokenize_and_align_labels_wrapper, batched=True)

    return datasets


def get_appended_datasets(
    retriever, tokenizer, all_doc_embeddings, doc_texts,
    train_agg_types=['naive', 'loose', 'strict', 'harsh', 'complex'],
    validate_agg_type='naive',
    max_tokens_total=512, max_tokens_each=250,
    top_k=1,
    cik_to_name=None
):
    def tokenize_and_align_labels_wrapper(examples):
        return tokenize_and_align_labels(examples, tokenizer=tokenizer)

    train_data, valid_data, test_data, expert_data = read_setting2_data()

    agg_train_keys = [f"{agg}_aggregation" for agg in train_agg_types]
    valid_key = f"{validate_agg_type}_aggregation"

    features = Features({
        'id': Value('string'),
        'tokens': Sequence(Value('string')),
        'texts': Value('string'),
        'labels': Sequence(Value('int32')),
        'highlight_probs': Sequence(Value('float32')),
        'aggregation': Value('string')
    })

    def make_dataset(data, agg_keys, split_name, expert=False):
        generator_fn = data_generator_for_expert if expert else data_generator_with_append
        try:
            if expert:
                expert_data_dict = {}
                for d in data:
                    if isinstance(d, dict):
                        expert_data_dict.update(d)
                generator_input = expert_data_dict.items()
            else:
                generator_input = data

            generated = []
            for g in generator_fn(
                generator_input,
                agg_keys,
                retriever,
                tokenizer,
                all_doc_embeddings,
                doc_texts,
                cik_to_name=cik_to_name,
                max_tokens_total=max_tokens_total,
                max_tokens_each=max_tokens_each,
                top_k=top_k
            ):
                generated.append(g)

            return Dataset.from_list(generated, features=features) if generated else Dataset.from_dict({'id': [], 'tokens': [], 'texts': [], 'labels': [], 'highlight_probs': [], 'aggregation': []})
        except Exception as e:
            return Dataset.from_dict({'id': [], 'tokens': [], 'texts': [], 'labels': [], 'highlight_probs': [], 'aggregation': []})

    datasets= {
        "train_append": make_dataset(train_data, agg_train_keys, 'train'),
        "valid_append": make_dataset(valid_data, [valid_key], 'valid'),
        "test_append": make_dataset(test_data, [valid_key], 'test'),
        "expert_append": make_dataset(expert_data, ['binary_labels'], 'expert', expert=True)
    }

    for key in ["train_append", "valid_append", "test_append"]:
        datasets[key] = datasets[key].map(tokenize_and_align_labels_wrapper, batched=True)
    
    return datasets


def data_generator_with_append(data_list, aggregation_labels, retriever, tokenizer, all_doc_embeddings, doc_texts, cik_to_name=None, max_tokens_total=512, max_tokens_each=250, top_k=1):
    sep_token = tokenizer.sep_token or "[SEP]"
    for sample in data_list:
        if not all(k in sample for k in ['tokens', 'highlight_probs', 'text']):
            continue

        tokens = sample['tokens']
        text = sample['text']
        highlight_probs = sample['highlight_probs']

        for key in aggregation_labels:
            if key in sample and 'label' in sample[key]:
                labels = sample[key]['label']
                agg_type = key.split('_')[0]

                try:
                    context = get_retrieved_context(sample, retriever, tokenizer, all_doc_embeddings, doc_texts, top_k=top_k, cik_to_name=cik_to_name)
                    context_tokens = tokenizer.tokenize(context)[:max_tokens_each]

                    truncated_tokens = tokens[:max_tokens_each]
                    truncated_labels = labels[:max_tokens_each]
                    truncated_probs = highlight_probs[:max_tokens_each]

                    combined_tokens = truncated_tokens + [sep_token] + context_tokens
                    combined_labels = truncated_labels + [-100] + [-100] * len(context_tokens)
                    combined_probs = truncated_probs + [0.0] + [0.0] * len(context_tokens)
                    combined_text = text + f" {sep_token} " + context

                    yield {
                        'id': sample.get('sample_id', 'unknown'),
                        'tokens': combined_tokens,
                        'texts': combined_text,
                        'labels': combined_labels,
                        'highlight_probs': combined_probs,
                        'aggregation': agg_type
                    }
                except Exception:
                    continue


def data_generator_for_expert(data_list, aggregation_labels, retriever, tokenizer, all_doc_embeddings, doc_texts, cik_to_name=None, max_tokens_total=512, max_tokens_each=250, top_k=1):
    sep_token = tokenizer.sep_token or "[SEP]"
    for sample_id, sample in data_list:
        if not all(k in sample for k in ['tokens', 'binary_labels']):
            continue

        text = sample.get('text') or sample.get('texts')
        if not text:
            continue

        tokens = sample['tokens'][:max_tokens_each]
        labels = sample['binary_labels'][:max_tokens_each]
        probs = [float(l) for l in labels]

        try:
            context = get_retrieved_context({'id': sample_id, 'text': text}, retriever, tokenizer, all_doc_embeddings, doc_texts, top_k=top_k, cik_to_name=cik_to_name)
            context_tokens = tokenizer.tokenize(context)[:max_tokens_each]

            combined_tokens = tokens + [sep_token] + context_tokens
            combined_labels = labels + [-100] + [-100] * len(context_tokens)
            combined_probs = probs + [0.0] + [0.0] * len(context_tokens)
            combined_text = text + f" {sep_token} " + context

            yield {
                'id': sample_id,
                'tokens': combined_tokens,
                'texts': combined_text,
                'labels': combined_labels,
                'highlight_probs': combined_probs,
                'aggregation': 'naive'
            }
        except Exception:
            continue


def get_retrieved_context(sample, retriever, tokenizer, all_doc_embeddings, doc_texts, top_k=1, cik_to_name=None):
    text = sample['text']
    if cik_to_name and 'id' in sample:
        try:
            query_firm_name = cik_to_name[sample['id'].split('_')[2]]
            text = f"{query_firm_name} {text}"
        except Exception:
            pass

    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(retriever.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = retriever(**inputs)
        query_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)

    query_embedding = query_embedding.unsqueeze(0)
    scores = torch.matmul(all_doc_embeddings, query_embedding.T).squeeze(1)
    top_indices = torch.topk(scores, k=top_k).indices.tolist()
    return " ".join([doc_texts[i] for i in top_indices])


def get_retrieved_context_no_firm_name(sample, retriever, tokenizer, all_doc_embeddings, doc_texts, top_k=1, cik_to_name=None):
    text = sample['text']
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(retriever.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = retriever(**inputs)
        query_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)

    query_embedding = query_embedding.unsqueeze(0)
    scores = torch.matmul(all_doc_embeddings, query_embedding.T).squeeze(1)
    top_indices = torch.topk(scores, k=top_k).indices.tolist()
    return " ".join([doc_texts[i] for i in top_indices])


if __name__ == '__main__':
    import argparse
    from datasets import DatasetDict, Dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_agg_type', '-t', nargs='+', default=['loose', 'strict'])
    parser.add_argument('--validation_agg_type', '-v', type=str, default='naive')
    train_data, valid_data, test_data, expert_data = read_setting2_data()
    args = parser.parse_args()

    train_agg_types = args.train_agg_type
    validate_agg_type = args.validation_agg_type

    agg_labels = [f"{agg}_aggregation" for agg in train_agg_types]
    # for united the input structure
    validate_agg_type = [f"{agg}_aggregation" for agg in validate_agg_type] 
    # validate_agg_type = f"{validate_agg_type}_aggregation"

    train_dataset = Dataset.from_generator(
        data_generator_mix_all,
        cache_dir='./data_cache',
        gen_kwargs={'data_list': train_data, 'aggregation_labels': agg_labels},
    )
    valid_dataset = Dataset.from_generator(
        data_generator_mix_all,
        cache_dir='./data_cache',
        gen_kwargs={'data_list': valid_data, 'aggregation_labels': [validate_agg_type]},
    )
    test_dataset = Dataset.from_generator(
        data_generator_mix_all,
        cache_dir='./data_cache',
        gen_kwargs={'data_list': test_data, 'aggregation_labels': [validate_agg_type]},
    )
    datasets = {"train": train_dataset, "valid": valid_dataset, "test": test_dataset}
    expert_dataset = Dataset.from_generator(
        data_generator_expert, gen_kwargs={'data_list': expert_data},
        cache_dir='./data_cache',
    )
    datasets["expert"] = expert_dataset
    dataset_dict = DatasetDict(datasets)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    def tokenize_and_align_labels_wrapper(examples):
        return tokenize_and_align_labels_cnc(examples, tokenizer=tokenizer)

    tokenized_datasets = dataset_dict.map(tokenize_and_align_labels_wrapper, batched=True)

    
    # tokenize_and_align_labels_cnc(train_dataset[0], tokenizer)
    # tokenize_and_align_labels(train_dataset[0], tokenizer)


