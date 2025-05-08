import os
import argparse
import time
from datetime import datetime
import numpy as np
import json
import random
from tqdm import tqdm
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define the start time and initialize time records
start_time = time.time()
time_records = {}


def run_testing_and_early_stopping(
    model, retriever, tokenizer, device,
    datasets, all_doc_embeddings, final_texts, cik_to_name,
    args, text_size, seed, checkpoint, retriever_training,
    train_agg_types, validate_agg_type
):
    best_earlystopping_score_epoch = 0 if args.early_stopping_metrics == 'f1' else 1

    testing_jobs = []

    for rank in [0, 1]:
        if rank == 0:
            testing_sets = {
                "training": datasets['train'],
                "validation": datasets['valid'],
                "testing": datasets['test'],
                "expert": datasets['expert']
            }
        else:
            appended_datasets = get_appended_datasets(
                retriever=retriever,
                tokenizer=tokenizer,
                all_doc_embeddings=all_doc_embeddings,
                doc_texts=final_texts,
                train_agg_types=train_agg_types,
                validate_agg_type=validate_agg_type,
                top_k=rank,
                cik_to_name=cik_to_name,
            )
            datasets.update(appended_datasets)
            testing_sets = {
                "training": datasets['train_append'],
                "validation": datasets['valid_append'],
                "testing": datasets['test_append'],
                "expert": datasets['expert_append']
            }
        
        for dataset_name, dataset in testing_sets.items():
            testing_jobs.append((rank, dataset_name, dataset))  # 把 rank、dataset_name、dataset 都存起來


    for rank, dataset_name, dataset in tqdm(testing_jobs, desc="Testing Progress"):
        tqdm.write(f"Testing: rank={rank}, dataset={dataset_name}")
        earlystopping_score = test(
            model=model,
            retriever=retriever,
            tokenizer=tokenizer,
            device=device,
            annotated_results=dataset,
            all_doc_embeddings=all_doc_embeddings,
            cik_to_name=cik_to_name,
            final_texts=final_texts,
            args=args,
            text_size=text_size,
            seed=seed,
            rank=rank,
            dataset_name=dataset_name,
            checkpoint=checkpoint,
            retriever_training=retriever_training,
            time_records=time_records,
            log_time_fn=log_time,
        )
        if args.early_stopping_metrics == 'f1' and earlystopping_score > best_earlystopping_score_epoch:
            best_earlystopping_score_epoch = earlystopping_score
        elif args.early_stopping_metrics == 'disorder' and earlystopping_score < best_earlystopping_score_epoch:
            best_earlystopping_score_epoch = earlystopping_score

    return best_earlystopping_score_epoch

def update_early_stopping(
    best_earlystopping_score_epoch,
    best_earlystopping_score,
    checkpoint,
    retriever_training,
    model,
    best_model,
    retriever,
    best_retriever,
    best_epoch,
    early_stopping_counter,
    early_stopping_update_times,
    args
):
    
    should_stop = False


    if args.early_stopping_metrics == 'f1':
        if best_earlystopping_score_epoch > best_earlystopping_score:
            best_earlystopping_score = best_earlystopping_score_epoch
            best_epoch = checkpoint * 2 + 1 + int(retriever_training)
            best_model = copy.deepcopy(model)
            store_best_model(args, best_model)
            best_retriever = copy.deepcopy(retriever)
            store_best_retriever(args, best_retriever)
            early_stopping_counter = 0
            early_stopping_update_times+=1
            log_time(f"Early stopping score ({args.early_stopping_metrics}), {early_stopping_update_times} times updated: {best_earlystopping_score} at epoch {best_epoch}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                should_stop = True

    elif args.early_stopping_metrics == 'disorder':
        if best_earlystopping_score_epoch < best_earlystopping_score:
            best_earlystopping_score = best_earlystopping_score_epoch
            best_epoch = checkpoint * 2 + 1 + int(retriever_training)
            best_model = copy.deepcopy(model)
            store_best_model(args, best_model)
            best_retriever = copy.deepcopy(retriever)
            store_best_retriever(args, best_retriever)
            early_stopping_counter = 0
            early_stopping_update_times+=1
            log_time(f"Early stopping score ({args.early_stopping_metrics}), {early_stopping_update_times} times updated: {best_earlystopping_score} at epoch {best_epoch}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                should_stop = True

    return best_earlystopping_score, best_epoch, best_model, best_retriever, early_stopping_counter, should_stop, early_stopping_update_times

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--training_size', type=int, help='Size of training data', default=800)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--retrieved_size', type=int, default=10)
    parser.add_argument('--retrieved_rank', '--rank', type=int, default=1)
    parser.add_argument('--seperate_type', '--sep', type=str, default='words')# words or tokens
    parser.add_argument('--least_relevant', type=bool, default=False)
    parser.add_argument('--training_time', type=int, default=10)
    parser.add_argument('--text_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--seed', '-s', type=int, default=42)
    parser.add_argument('--checkpoint','-c', type=int, default=-1)
    parser.add_argument('--testnum','-tn', type=int, required=True, help='Test number is the same as the RAGTest number')
    parser.add_argument('--early_stopping_metrics','-es', type=str, choices=['f1', 'disorder'], required=True, help='Metrics for early stopping: f1 or disorder')
    parser.add_argument('--training_label','-tl', type=str, choices=['strict', 'naive', 'loose'], required=True, help='Label for training: naive, strict, or loose')
    parser.add_argument('--validation_label','-vl', type=str, choices=['strict', 'naive', 'loose'], default='naive', help='Label for validation: naive, strict, or loose')
    parser.add_argument('--append_to_training','-at', type=bool, default=True, help='Whether to append the training set with the retrieved data')
    parser.add_argument('--append_rank_to_training','-art', type=int, default=1, help='Rank of appended data to training set')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    import torch
    import pickle
    import copy
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel, BertForTokenClassification
    from transformers import DataCollatorForTokenClassification
    from transformers.data.data_collator import DataCollatorMixin, pad_without_fast_tokenizer_warning
    from train_highlighter import train_highlighter
    from train_retriever import train_retriever
    from Ragtest import test
    from datasets import DatasetDict
    from def_functions import (
        test_memory_allocation,
        store_model,
        store_retriever,
        InfoNCELoss,
        embed_texts_contriever2,
        store_best_model,
        store_best_retriever,
    )
    from data_utils import(
        read_setting2_data,
        get_raw_datasets,
        get_appended_datasets,
)

    
    # Seed for reproducibility
    seed=args.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


    testnum=args.testnum
    text_size =  args.text_size
    BATCH_SIZE= args.batch_size
    checkpoint=args.checkpoint

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    rank_list=[0,1]

    cik_to_name = {}

    directory1 = "/home/ythsiao/output"
    firm_list = os.listdir(directory1)

    for firm in firm_list:
        directory2 = os.path.join(directory1, firm)
        directory3 = os.path.join(directory2, "10-K")
        tenK_list = os.listdir(directory3)
        tenK_list = sorted(tenK_list)


        for tenK in tenK_list:
            if tenK[:4] == "2022":
                file_path = os.path.join(directory3, tenK)
                with open(file_path, 'r') as file:
                    for line in file:
                        data = json.loads(line)
                        if ('company_name' in data) and ('cik' in data):
                            firm_name = data['company_name']
                            cik = data['cik']
                            cik_to_name[cik] = firm_name

    with open('/home/yhwang/Fin_highlighter/para_info/para_info_contriever_firm.pkl', 'rb') as f:
            para_info = pickle.load(f)

    time_output_path = f'/home/teyu/Fin_highlighter_new/time_ana/{testnum}/{seed}'
    os.makedirs(time_output_path, exist_ok=True)

    # determine the base filename and extension
    base_filename = "time_records"
    extension = ".json"
    filename = base_filename + extension
    filepath = os.path.join(time_output_path, filename)
    counter = 1

    # determine a unique filename
    while os.path.exists(filepath):
        filename = f"{base_filename}_{counter}{extension}"
        filepath = os.path.join(time_output_path, filename)
        counter += 1
    
    def log_time(event_name):
        elapsed = time.time() - start_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        time_records[event_name] = {
            "elapsed_time": f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}",
            "system_time": now
        }

        print(f"[Timer] {event_name}: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} (System time: {now})")

        with open(filepath, "w") as f:
            json.dump(time_records, f, indent=4)
    log_time('Start time')

    # Extract embeddings and texts
    final_embeddings = np.vstack([item[2] for item in para_info]).astype('float32')
    final_texts = [item[1] for item in para_info]
    final_ids = [item[3] for item in para_info]

    # Initialize tokenizer and retriever model
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    retriever = AutoModel.from_pretrained('facebook/contriever')
    retriever.to(device).train()

    # Initialize classification model
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device).train()


    train_agg_types=[]
    train_agg_types.append(args.training_label)
    validate_agg_type=args.validation_label

    # get the dataset and transform to dictionary
    datasets = get_raw_datasets(
            train_agg_types=train_agg_types, 
            validate_agg_type=validate_agg_type,
            tokenizer=tokenizer,
        )
    all_doc_embeddings = None
    # Do the pre-work for the new training
    print(f'checkpoint:{checkpoint}')
    if checkpoint == -1:

        # Convert embeddings to tensors initially
        checkpoint=0
        all_doc_embeddings = torch.tensor(final_embeddings).to(device)  # Shape: (num_docs, embedding_dim)


        # Test for pre-trained highlighter and retriever
        for rank in rank_list:
            if rank == 0:
                for dataset_name, dataset in {"training": datasets['train'], "validation": datasets['valid'], "testing": datasets['test'], "expert":datasets['expert']}.items():
                    test(model, retriever, tokenizer, device, dataset, all_doc_embeddings,
                            cik_to_name, final_texts, args, text_size, seed, rank, dataset_name, retriever_training=True, checkpoint=-1)
            else:
                appended_datasets= get_appended_datasets(
                    retriever=retriever, 
                    tokenizer=tokenizer, 
                    all_doc_embeddings=all_doc_embeddings, 
                    doc_texts=final_texts,
                    train_agg_types=train_agg_types, 
                    validate_agg_type=validate_agg_type,
                    top_k=rank,
                    cik_to_name=cik_to_name,
                )
                datasets.update(appended_datasets)

                for dataset_name, dataset in {"training": datasets['train_append'], "validation": datasets['valid_append'], "testing": datasets['test_append'], "expert":datasets['expert_append']}.items():
                    test(model, retriever, tokenizer, device, dataset, all_doc_embeddings,
                            cik_to_name, final_texts, args, text_size, seed, rank, dataset_name, retriever_training=True,checkpoint=-1)
    
        log_time('zero shot test finished!')
        

    else:
        try:
            dir_path=f'/home/teyu/Fin_highlighter_new/rag_model/{testnum}/{seed}/retriever'
            # checkpoint = directly find the highest checkpoint of the retriever in the directory
            # if assigned checkpoint is bigger then the biggest checkpoint in the directory, then use the biggest checkpoint
            print(f'check checkpoint from retriever from {dir_path}')
            checkpoint = min(max([int(f.split('_')[2]) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.startswith('retriever') and f.endswith('.pth')]),args.checkpoint)
            # load the model with the checkpoint
            print(f'the checkpoint: checkpoint {checkpoint}')
            model.load_state_dict(torch.load(f'/home/teyu/Fin_highlighter_new/rag_model/{testnum}/{seed}/highlighter/model_checkpoint_{checkpoint}_1.pth', map_location=device))
            print(f'loaded model from checkpoint success')
            retriever.load_state_dict(torch.load(f'/home/teyu/Fin_highlighter_new/rag_model/{testnum}/{seed}/retriever/retriever_checkpoint_{checkpoint}_1.pth', map_location=device))
            print(f'loaded retriever from checkpoint success')
            retriever.eval()
            model.eval()
            
            batch_size = 512  # Define the desired batch size
            all_doc_embeddings = []
            
            # Iterate over final_texts in chunks of batch_size
            for i in range(0, len(final_texts), batch_size):
                print(f'{i}/{len(final_texts)}')
                batch = final_texts[i:i + batch_size]
                batch_embeddings = embed_texts_contriever2(batch, retriever, tokenizer, device)
                while True:
                    try:
                        all_doc_embeddings.append(batch_embeddings)
                        break
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print("CUDA out of memory. Waiting 10 seconds and clearing cache...")
                            torch.cuda.empty_cache()
                            time.sleep(10)
                        else:
                            raise e  
            all_doc_embeddings = torch.cat(all_doc_embeddings, dim=0)
            all_doc_embeddings = F.normalize(all_doc_embeddings, p=2, dim=1)

        except:
            checkpoint=0
            all_doc_embeddings = torch.tensor(final_embeddings).to(device)  # Shape: (num_docs, embedding_dim)
            log_time('loading error!')
            raise Exception
            

    appended_datasets= get_appended_datasets(
            retriever=retriever, 
            tokenizer=tokenizer, 
            all_doc_embeddings=all_doc_embeddings, 
            doc_texts=final_texts,
            train_agg_types=train_agg_types, 
            validate_agg_type=validate_agg_type
        )

    datasets.update(appended_datasets)
    dataset_dict = DatasetDict(datasets)

    # Setup optimizer and loss criterion
    optimizer1 = torch.optim.Adam(
        list(model.parameters()), lr=args.lr
    )
    optimizer2 = torch.optim.Adam(
        list(retriever.parameters()), lr=args.lr
    )
    criterion1 = torch.nn.BCELoss()
    criterion2 = InfoNCELoss(temperature=0.05)


    #define early stopping variable
    if checkpoint==0:
        if args.early_stopping_metrics=='f1':
            best_earlystopping_score=0
        elif args.early_stopping_metrics=='disorder':
            best_earlystopping_score=1
        best_epoch=0
        best_model=None
        best_retriever=None
        early_stopping_counter=0
        early_stopping_update_times=0
    else:
        best_earlystopping_score=0.46622330379509663
        best_epoch=11
        best_model=model.load_state_dict(torch.load(f'/home/teyu/Fin_highlighter_new/rag_model/{testnum}/{seed}/highlighter/model_checkpoint_7_1.pth', map_location=device))
        best_retriever= retriever.load_state_dict(torch.load(f'/home/teyu/Fin_highlighter_new/rag_model/{testnum}/{seed}/retriever/retriever_checkpoint_7_1.pth', map_location=device))
        early_stopping_counter=2
        early_stopping_update_times=6
        print(f'early stopping counter: {early_stopping_counter}')
        print(f'early stopping update times: {early_stopping_update_times}')
    
    while True:
        

        model.train()
        retriever.eval()
        retriever_training=False
        
        log_time(f'The {checkpoint+1} time training highlighter')
        # Train the highlighter
        model_trainer=train_highlighter(
            args=args,
            model=model,
            retriever=retriever,
            tokenizer=tokenizer,
            train_agg_types=train_agg_types,
            validate_agg_type=validate_agg_type,
            metric_for_best_model=f'eval_train_{args.early_stopping_metrics}',
            greater_is_better= True if args.early_stopping_metrics=='f1' else False,
            append_to_training=args.append_to_training,
            seed=args.seed,
            device=device,
            all_doc_embeddings=all_doc_embeddings,
            top_k=args.append_rank_to_training,
            cik_to_name=cik_to_name,
            final_texts=final_texts,
        )
        model=model_trainer.model

        log_time(f'The {checkpoint+1} time training highlighter finished! start testing')

        store_model(args, model, checkpoint)
        
        # Test
        best_earlystopping_score_epoch = run_testing_and_early_stopping(
            model=model,
            retriever=retriever,
            tokenizer=tokenizer,
            device=device,
            datasets=datasets,
            all_doc_embeddings=all_doc_embeddings,
            final_texts=final_texts,
            cik_to_name=cik_to_name,
            args=args,
            text_size=text_size,
            seed=seed,
            checkpoint=checkpoint,
            retriever_training=retriever_training,
            train_agg_types=train_agg_types,
            validate_agg_type=validate_agg_type,
        )

        best_earlystopping_score, best_epoch, best_model, best_retriever, early_stopping_counter, should_stop, early_stopping_update_times = update_early_stopping(
            best_earlystopping_score_epoch=best_earlystopping_score_epoch,
            best_earlystopping_score=best_earlystopping_score,
            checkpoint=checkpoint,
            retriever_training=retriever_training,
            model=model,
            best_model=best_model,
            retriever=retriever,
            best_retriever=best_retriever,
            best_epoch=best_epoch,
            early_stopping_counter=early_stopping_counter,
            early_stopping_update_times=early_stopping_update_times,
            args=args,
        )
        log_time(f'Test finished. The {checkpoint+1} time training highlighter finished!')
        if should_stop:
            break
       
        log_time(f'The {checkpoint+1} time training retriever')
        # Train the retriever
        retriever.train()
        model.eval()
        retriever_training=True
        
        retriever, all_doc_embeddings =train_retriever(
            model=model,
            retriever=retriever,
            tokenizer=tokenizer,
            training_annotated_results=datasets['train'],
            final_texts=final_texts,
            cik_to_name=cik_to_name,
            device=device,
            text_size=text_size,
            batch_size=BATCH_SIZE,
            patience=args.patience,
            args=args,
            time_records=time_records,
            log_time_fn=log_time,
            checkpoint=checkpoint,
        )
        log_time(f'The {checkpoint+1} time training retriever finished! start testing')
        store_retriever(args, retriever, checkpoint)

        # Test
        best_earlystopping_score_epoch = run_testing_and_early_stopping(
            model=model,
            retriever=retriever,
            tokenizer=tokenizer,
            device=device,
            datasets=datasets,
            all_doc_embeddings=all_doc_embeddings,
            final_texts=final_texts,
            cik_to_name=cik_to_name,
            args=args,
            text_size=text_size,
            seed=seed,
            checkpoint=checkpoint,
            retriever_training=retriever_training,
            train_agg_types=train_agg_types,
            validate_agg_type=validate_agg_type,
        )

        best_earlystopping_score, best_epoch, best_model, best_retriever, early_stopping_counter, should_stop, early_stopping_update_times = update_early_stopping(
            best_earlystopping_score_epoch=best_earlystopping_score_epoch,
            best_earlystopping_score=best_earlystopping_score,
            checkpoint=checkpoint,
            retriever_training=retriever_training,
            model=model,
            best_model=best_model,
            retriever=retriever,
            best_retriever=best_retriever,
            best_epoch=best_epoch,
            early_stopping_counter=early_stopping_counter,
            early_stopping_update_times=early_stopping_update_times,
            args=args,
        )


       
        log_time(f'Test finished. The {checkpoint+1} time training retriever finished!')

        if should_stop:
            break
        checkpoint +=1
    

    log_time('Iterative training finished!')
    print(f"Early stopping at epoch {checkpoint*2+1+int(retriever_training)} with best {args.early_stopping_metrics} score: {best_earlystopping_score}")
    print(f"Best model saved at epoch {best_epoch} with {args.early_stopping_metrics} score: {best_earlystopping_score}")
    print(f"Best retriever saved at epoch {best_epoch} with {args.early_stopping_metrics} score: {best_earlystopping_score}")
    
    
    
    

    print("Training finished!")

    

