from dataset import UMLSDataset, my_collate_fn
from model import UMLSFinetuneModel
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from tqdm import tqdm, trange
import torch
from torch import nn
import time
import os
import numpy as np
import argparse
import time
import pathlib
from torch.utils.data import DataLoader
import pickle
from torch.utils.tensorboard import SummaryWriter
from test_faiss import find_new_index
from accelerate import Accelerator
#import ipdb
# try:
#     from torch.utils.tensorboard import SummaryWriter
# except:
# from tensorboardX import SummaryWriter


def train(args, model, train_dataloader, loss_list):
    writer = SummaryWriter(comment='umls')

    t_total = args.max_steps

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    
    if args.use_multi_gpu:
        model, optimizer, train_dataloader = args.accelerator.prepare(model, optimizer, train_dataloader)

    args.warmup_steps = int(args.warmup_steps)
    if args.schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
    if args.schedule == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )
    if args.schedule == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

    args.accelerator.print("***** Running training *****")
    args.accelerator.print("  Total Steps =", t_total)
    args.accelerator.print("  Steps needs to be trained=", t_total - args.shift)
    args.accelerator.print("  Instantaneous batch size per GPU =", args.train_batch_size)
    args.accelerator.print(
        "  Total train batch size (w. parallel, distributed & accumulation) =",
        args.train_batch_size
        * args.gradient_accumulation_steps * args.nr_gpus,
    )
    args.accelerator.print("  Gradient Accumulation steps =", args.gradient_accumulation_steps)

    model.zero_grad()

    for i in range(args.shift):
        scheduler.step()
    global_step = args.shift

    while True:
        model.train()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", ascii=True, disable=not args.accelerator.is_local_main_process)
        batch_loss = 0.
        batch_sty_loss = 0.
        batch_cui_loss = 0.
        batch_re_loss = 0.
        for _, batch in enumerate(epoch_iterator):
            input_ids = batch[0].to(args.device)
            cui_label = batch[1].to(args.device)
            attention_mask = batch[2].to(args.device)
            # for item in batch:
            #     print(item.shape)

            loss = model(input_ids, cui_label, attention_mask)
            batch_loss = float(loss.item())
            loss_list.append(batch_loss)
            # tensorboardX
            writer.add_scalar('batch_loss', batch_loss,
                              global_step=global_step)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            

            args.accelerator.backward(loss)

            epoch_iterator.set_description("Loss: %0.4f" % batch_loss)

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
            del input_ids
            del cui_label
            del attention_mask
            global_step += 1
            if global_step % args.save_step == 0 and global_step > 0:
                save_path = os.path.join(
                    args.output_dir, f'model_{global_step}.pth')
                unwrapped_model = args.accelerator.unwrap_model(model)
                torch.save(unwrapped_model.bert, save_path)
                with open(os.path.join(args.output_dir, f'loss_{global_step}.pkl'), 'wb') as f:
                    pickle.dump(loss_list, f)
                torch.cuda.empty_cache()
            if args.max_steps > 0 and global_step % args.faiss_step == 0:
                return None

    return None


def run(args):
    torch.manual_seed(args.seed)  # cpu
    torch.cuda.manual_seed(args.seed)  # gpu
    np.random.seed(args.seed)  # numpy
    torch.backends.cudnn.deterministic = True  # cudnn

    if os.path.exists(args.output_dir):
        indices_list = []
        for f in os.listdir(args.output_dir):
            if f[0:7] == "indices" and f[-4:] == ".npy":
                indices_list.append(int(f[8:-4]))
        if len(indices_list) > 0:
            i = max(indices_list)
            args.indices_path = os.path.join(args.output_dir, f'indices_{i}.npy')
            i += 1
        else:
            i = 0
    else:
        i = 0
    #args.output_dir = args.output_dir + "_" + str(int(time.time()))
    while i < (int(args.max_steps / args.faiss_step)):
        print('i', i)
        args.accelerator = Accelerator()
        args.device = args.accelerator.device
        # dataloader
        if args.lang == "eng":
            lang = ["ENG"]
        if args.lang == "all":
            lang = None
            assert args.model_name_or_path.find("bio") == -1, "Should use multi-language model"
        args.accelerator.print(args.indices_path)
        umls_dataset = UMLSDataset(umls_folder=args.umls_dir, 
                                model_name_or_path=args.model_name_or_path, 
                                idx2phrase_path=args.idx2phrase_path, 
                                phrase2idx_path=args.phrase2idx_path,
                                indices_path=args.indices_path,
                                max_length=args.max_seq_length)
        umls_dataloader = DataLoader(umls_dataset,
                                    batch_size=args.train_batch_size, 
                                    shuffle=True,
                                    num_workers=args.num_workers, 
                                    pin_memory=True, 
                                    drop_last=True,
                                    collate_fn=my_collate_fn)
        model_load = False
        if os.path.exists(args.output_dir):
            save_list = []
            for f in os.listdir(args.output_dir):
                if f[0:5] == "model" and f[-4:] == ".pth":
                    save_list.append(int(f[6:-4]))
            if len(save_list) > 0:
                args.shift = max(save_list)
                with open(os.path.join(args.output_dir, 'loss_'+str(args.shift)+'.pkl'), 'rb') as f:
                    loss_list = pickle.load(f)
                model = UMLSFinetuneModel(device=args.device, 
                                    model_name_or_path=args.model_name_or_path, 
                                    cui_label_count=len(umls_dataset.cui2idx)).to(args.device)
                if os.path.exists(os.path.join(args.output_dir, 'last_model.pth')):
                    model.bert = torch.load(os.path.join(
                        args.output_dir, 'last_model.pth')).to(args.device)
                    model_load = True
                    args.accelerator.print('load last model')
                else:
                    model.bert = torch.load(os.path.join(
                        args.output_dir, f'model_{max(save_list)}.pth')).to(args.device)
                    model_load = True
                    args.accelerator.print('load model{}'.format(max(save_list)))
        if not model_load:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            model = UMLSFinetuneModel(device=args.device, 
                                    model_name_or_path=args.model_name_or_path, 
                                    cui_label_count=len(umls_dataset.cui2idx)).to(args.device)
            args.shift = 0
            model_load = True
            loss_list = []
        
        if args.do_train:
            torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
            train(args, model, umls_dataloader, loss_list)
            args.accelerator.wait_for_everyone()
            unwrapped_model = args.accelerator.unwrap_model(model)
            args.accelerator.save(unwrapped_model.bert, os.path.join(args.output_dir, 'last_model.pth'))
            del model
            torch.cuda.empty_cache()
            new_CODER_path = os.path.join(args.output_dir, 'last_model.pth')
            new_indices_path = os.path.join(args.output_dir, f'indices_{i}.npy')
            new_similarity_path = os.path.join(args.output_dir, f'similarity_{i}.npy')
            if args.accelerator.is_main_process:
                find_new_index(new_CODER_path, new_indices_path, new_similarity_path)
            args.accelerator.wait_for_everyone()
            args.indices_path = new_indices_path
        i += 1
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--umls_dir",
        default="../umls",
        type=str,
        help="UMLS dir",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="GanjinZero/UMLSBert_ENG",
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: ",
    )
    parser.add_argument(
        "--idx2phrase_path",
        default="data/idx2string.pkl",
        type=str,
        help="Path to dict index: phrase"
    )
    parser.add_argument(
        "--phrase2idx_path",
        default="data/string2idx.pkl",
        type=str,
        help="Path to dict phrase: index"
    )
    parser.add_argument(
        "--indices_path",
        default="data/indices.npy",
        type=str,
        help="Path to indices obtained by faiss"
    )
    parser.add_argument(
        "--output_dir",
        default="output_30pos",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--faiss_step",
        default=100000,
        type=int,
        help="Faiss step",
    )
    parser.add_argument(
        "--save_step",
        default=25000,
        type=int,
        help="Save step",
    )

    # Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=32,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", default=True, type=bool, help="Whether to run training.")
    parser.add_argument(
        "--train_batch_size", default=4, type=int, help="Batch size of groups per GPU/CPU for training. A group contains 32 strings (current + top30 + same cui).",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=2e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01,
                        type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8,
                        type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0,
                        type=float, help="Max gradient norm.")
    parser.add_argument(
        "--max_steps",
        default=400000,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=40000,
                        help="Linear warmup over warmup_steps or a float.")
    parser.add_argument("--device", type=str, default='cuda:1', help="device")
    parser.add_argument("--seed", type=int, default=72,
                        help="random seed for initialization")
    parser.add_argument("--schedule", type=str, default="linear",
                        choices=["linear", "cosine", "constant"], help="Schedule.")
    parser.add_argument("--num_workers", default=1, type=int,
                        help="Num workers for data loader, only 0 can be used for Windows")
    parser.add_argument("--lang", default='eng', type=str, choices=["eng", "all"],
                        help="language range, eng or all")

    parser.add_argument("--use_multi_gpu", type=bool, default=False, help="Use multi-gpu or not")
    parser.add_argument("--nr_gpus", type=int, default=2, help="number of gpus")
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
