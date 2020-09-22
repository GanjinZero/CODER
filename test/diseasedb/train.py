import sys
sys.path.append("../../pretrain/")
from linear_model import LinearModel
from load_umls import UMLS
import numpy as np
import os
import shutil
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup, AutoConfig, AutoModel
from time import time
from tqdm import tqdm
import ipdb


# parameters
embedding = sys.argv[1]
embedding_type = sys.argv[2]
freeze_embedding = sys.argv[3]
device = sys.argv[4]

if freeze_embedding.lower() in ['t', 'true']:
    freeze_embedding = True
else:
    freeze_embedding = False

if device == "0":
    device = "cuda:0"
if device == "1":
    device = "cuda:1"

if embedding_type == 'bert':
    epoch_num = 50
    if freeze_embedding:
        batch_size = 512
        learning_rate = 1e-3
    else:
        batch_size = 96
        learning_rate = 2e-5
    max_seq_length = 32
    try:
        tokenizer = AutoTokenizer.from_pretrained(embedding)
    except BaseException:
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(embedding, "../"))
else:
    epoch_num = 50
    batch_size = 512
    learning_rate = 1e-3
    max_seq_length = 16

def pad(l):
    if len(l) > max_seq_length:
        return l[0:max_seq_length]
    return l + [use_embedding_count - 1] * (max_seq_length - len(l))

# load train and test
cui_train_0 = []
cui_train_1 = []
rel_train = []
with open("./data/x_train.txt") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split("\t")
        cui_train_0.append(line[0])
        cui_train_1.append(line[1])
with open("./data/y_train.txt") as f:
    lines = f.readlines()
    for line in lines:
        rel_train.append(line.strip())

cui_test_0 = []
cui_test_1 = []
rel_test = []
with open("./data/x_test.txt") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split("\t")
        cui_test_0.append(line[0])
        cui_test_1.append(line[1])
with open("./data/y_test.txt") as f:
    lines = f.readlines()
    for line in lines:
        rel_test.append(line.strip())

# build rel2id
rel_set = set(rel_train + rel_test)
rel2id = {rel: index for index, rel in enumerate(list(rel_set))}
id2rel = {index: rel for rel, index in rel2id.items()}
cui_set = set(cui_train_0 + cui_train_1 + cui_test_0 + cui_test_1)
print('Count of differnt cui:', len(cui_set))

# Deal cui type embedding
if embedding_type != 'bert':
    if embedding.find('txt') >= 0:
        with open(embedding, "r", encoding="utf-8") as f:
            line = f.readline()
            count, dim = map(int, line.strip().split())
            lines = f.readlines()

if embedding_type == 'cui':
    # build cui2id and use_embedding
    if embedding.find('txt') >= 0:
        cui2id = {}
        use_embedding_count = 0
        emb_sum = np.zeros(shape=(dim), dtype=float)
        for line in lines:
            l = line.strip().split()
            cui = l[0]
            if embedding.find('stanford') >= 0:
                cui = 'C' + cui
            emb = np.array(list(map(float, l[1:])))
            emb_sum += emb
            if cui in cui_set:
                cui2id[cui] = use_embedding_count
                if use_embedding_count == 0:
                    use_embedding = emb
                else:
                    use_embedding = np.concatenate((use_embedding, emb), axis=0)
                use_embedding_count += 1
        emb_avg = emb_sum / use_embedding_count
        use_embedding = np.concatenate((use_embedding, emb_avg), axis=0)
        use_embedding_count += 1
        use_embedding = use_embedding.reshape((-1, dim))
        print('Embedding shape:', use_embedding.shape)
    if embedding.find('pkl') >= 0:
        import pickle
        with open(embedding, 'rb') as f:
            W = pickle.load(f)
        cui2id = {}
        use_embedding_count = 0
        dim = len(list(W.values())[0][1:-1].split(','))
        emb_sum = np.zeros(shape=(dim), dtype=float)
        for cui in cui_set:
            if cui in W and not cui in cui2id:
                emb = np.array([float(num) for num in W[cui][1:-1].split(',')])
                #ipdb.set_trace()
                emb_sum += emb
                cui2id[cui] = use_embedding_count
                if use_embedding_count == 0:
                    use_embedding = emb
                else:
                    use_embedding = np.concatenate((use_embedding, emb), axis=0)
                use_embedding_count += 1
        emb_avg = emb_sum / use_embedding_count
        if 'empty' in W:
            emb_avg = np.array([float(num) for num in W['empty'][1:-1].split(',')])
        use_embedding = np.concatenate((use_embedding, emb_avg), axis=0)
        use_embedding_count += 1
        use_embedding = use_embedding.reshape((-1, dim))
        print('Embedding shape:', use_embedding.shape)

    # apply cui2id and rel2id
    train_input_0 = [cui2id.get(cui, use_embedding_count - 1)
                     for cui in cui_train_0]
    train_input_1 = [cui2id.get(cui, use_embedding_count - 1)
                     for cui in cui_train_1]
    train_y = [rel2id[rel] for rel in rel_train]
    test_input_0 = [cui2id.get(cui, use_embedding_count - 1)
                    for cui in cui_test_0]
    test_input_1 = [cui2id.get(cui, use_embedding_count - 1)
                    for cui in cui_test_1]
    test_y = [rel2id[rel] for rel in rel_test]

# Find standard term name
if not embedding_type == 'cui':
    umls = UMLS("../../umls", only_load_dict=True)
    cui2str = {}
    #ipdb.set_trace()
    for cui in cui_set:
        standard_term = umls.search(code=cui, max_number=1)
        if standard_term is not None:
            cui2str[cui] = standard_term[0]
        else:
            cui2str[cui] = cui

# Deal word type embedding
if embedding_type == 'word':

    # tokenize
    from nltk.tokenize import word_tokenize
    cui2tokenize = {}
    for cui in cui2str:
        cui2tokenize[cui] = word_tokenize(cui2str[cui])

    # build word2id and use_embedding
    word2id = {}
    use_embedding_count = 0

    if embedding.find('txt') >= 0:
        emb_sum = np.zeros(shape=(dim), dtype=float)
        for line in lines:
            l = line.strip().split()
            word = l[0]
            emb = np.array(list(map(float, l[1:])))
            emb_sum += emb
            word2id[word] = use_embedding_count
            if use_embedding_count == 0:
                use_embedding = emb
            else:
                use_embedding = np.concatenate((use_embedding, emb), axis=0)
            use_embedding_count += 1
        emb_avg = emb_sum / use_embedding_count
        use_embedding = np.concatenate((use_embedding, emb_avg), axis=0)
        use_embedding_count += 1
        emb_zero = np.zeros_like(emb_avg)
        use_embedding = np.concatenate((use_embedding, emb_zero), axis=0)
        use_embedding_count += 1
        use_embedding = use_embedding.reshape((-1, dim))
        print('Embedding shape:', use_embedding.shape)
    if embedding.find('bin') >= 0:
        import gensim
        model = gensim.models.KeyedVectors.load_word2vec_format(embedding, binary=True)
        emb_sum = np.zeros(shape=(model.vector_size), dtype=float)
        for cui in cui2tokenize:
            for w in cui2tokenize[cui]:
                if w in model and not w in word2id:
                    emb = model[w]
                    emb_sum += emb
                    word2id[w] = use_embedding_count
                    if use_embedding_count == 0:
                        use_embedding = emb
                    else:
                        use_embedding = np.concatenate((use_embedding, emb), axis=0)
                    use_embedding_count += 1  
        emb_avg = emb_sum / use_embedding_count
        use_embedding = np.concatenate((use_embedding, emb_avg), axis=0)
        use_embedding_count += 1
        emb_zero = np.zeros_like(emb_avg)
        use_embedding = np.concatenate((use_embedding, emb_zero), axis=0)
        use_embedding_count += 1
        use_embedding = use_embedding.reshape((-1, model.vector_size))
        print('Original embedding count:', len(model.wv.vocab))
        print('Embedding shape:', use_embedding.shape)                  

    # apply word2id and rel2id
    train_input_0 = [[word2id[w] for w in cui2tokenize[cui] if w in word2id] for cui in cui_train_0]
    train_input_1 = [[word2id[w] for w in cui2tokenize[cui] if w in word2id] for cui in cui_train_1]
    train_y = [rel2id[rel] for rel in rel_train]
    test_input_0 = [[word2id[w] for w in cui2tokenize[cui] if w in word2id] for cui in cui_test_0]
    test_input_1 = [[word2id[w] for w in cui2tokenize[cui] if w in word2id] for cui in cui_test_1]
    test_y = [rel2id[rel] for rel in rel_test]

    # average and padding
    # deal with input length = 0, use average
    train_input_0 = [cui if cui else [use_embedding_count - 2] for cui in train_input_0] 
    train_input_1 = [cui if cui else [use_embedding_count - 2] for cui in train_input_1]
    test_input_0 = [cui if cui else [use_embedding_count - 2] for cui in test_input_0]
    test_input_1 = [cui if cui else [use_embedding_count - 2] for cui in test_input_1]
    # calculate length
    train_length_0 = [len(cui) for cui in train_input_0]
    train_length_1 = [len(cui) for cui in train_input_1]
    test_length_0 = [len(cui) for cui in test_input_0]
    test_length_1 = [len(cui) for cui in test_input_1]
    # padding
    train_input_0 = list(map(pad, train_input_0))
    train_input_1 = list(map(pad, train_input_1))
    test_input_0 = list(map(pad, test_input_0))
    test_input_1 = list(map(pad, test_input_1))

# Deal bert type embedding
if embedding_type == 'bert':
    train_input_0 = []
    train_input_1 = []
    test_input_0 = []
    test_input_1 = []

    cui2tokenize = {}
    for cui in cui2str:
        cui2tokenize[cui] = tokenizer.encode_plus(
            cui2str[cui], max_length=max_seq_length, add_special_tokens=True,
            truncation=True, pad_to_max_length=True)['input_ids']
    
    train_input_0 = [cui2tokenize[cui] for cui in cui_train_0]
    train_input_1 = [cui2tokenize[cui] for cui in cui_train_1]
    test_input_0 = [cui2tokenize[cui] for cui in cui_test_0]
    test_input_1 = [cui2tokenize[cui] for cui in cui_test_1]
    train_y = [rel2id[rel] for rel in rel_train]
    test_y = [rel2id[rel] for rel in rel_test]

# Dataset and Dataloader
train_input_0 = torch.LongTensor(train_input_0)
train_input_1 = torch.LongTensor(train_input_1)
test_input_0 = torch.LongTensor(test_input_0)
test_input_1 = torch.LongTensor(test_input_1)
train_y = torch.LongTensor(train_y)
test_y = torch.LongTensor(test_y)
if embedding_type != 'word':
    train_dataset = TensorDataset(train_input_0, train_input_1, train_y)
    test_dataset = TensorDataset(test_input_0, test_input_1, test_y)
else:
    train_length_0 = torch.LongTensor(train_length_0)
    train_length_1 = torch.LongTensor(train_length_1)
    test_length_0 = torch.LongTensor(test_length_0)
    test_length_1 = torch.LongTensor(test_length_1)
    train_dataset = TensorDataset(train_input_0, train_input_1, train_length_0, train_length_1, train_y)
    test_dataset = TensorDataset(test_input_0, test_input_1, test_length_0, test_length_1, test_y)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

# Prepare model and optimizier
# model
if embedding_type != 'bert':
    use_embedding = torch.FloatTensor(use_embedding)
    model = LinearModel(len(rel2id), embedding_type, use_embedding, freeze_embedding).to(device)
else:
    try:
        config = AutoConfig.from_pretrained(embedding)
        bert_model = AutoModel.from_pretrained(embedding, config=config).to(device)
    except BaseException:
        bert_model = torch.load(os.path.join(embedding, 'pytorch_model.bin')).to(device)
    model = LinearModel(len(rel2id), embedding_type, bert_model, freeze_embedding).to(device)

# optimizier
if embedding_type != 'bert':
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
if embedding_type == "bert":
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(epoch_num * len(train_dataloader) * 0.1),
                                                num_training_steps=epoch_num * len(train_dataloader))

# Prepare eval function
from sklearn.metrics import accuracy_score, classification_report, f1_score
def eval(m, dataloader):
    y_pred = []
    y_true = []
    m.eval()
    with torch.no_grad():
        for batch in dataloader:
            x0 = batch[0].to(device)
            x1 = batch[1].to(device)
            if m.embedding_type == "word":
                l0 = batch[2].to(device)
                l1 = batch[3].to(device)
                r = batch[4]
            else:
                l0 = l1 = None
                r = batch[2]
            pred, loss = m(x0, x1, l0, l1)
            y_pred += torch.max(pred, dim=1)[1].detach().cpu().numpy().tolist()
            y_true += r.detach().cpu().numpy().tolist()
    acc = accuracy_score(y_true, y_pred) * 100
    #f1 = f1_score(y_true, y_pred) * 100
    report = classification_report(y_true, y_pred)
    return acc, report

# Train and eval
if not os.path.exists("./result/"):
    os.mkdir("./result/")

for epoch_index in range(epoch_num):
    model.train()
    epoch_loss = 0.
    time_now = time()
    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        x0 = batch[0].to(device)
        x1 = batch[1].to(device)
        if model.embedding_type == "word":
            l0 = batch[2].to(device)
            l1 = batch[3].to(device)
            r = batch[4].to(device)
        else:
            l0 = l1 = None
            r = batch[2].to(device)
        pred, loss = model(x0, x1, l0, l1, r)
        loss.backward()
        optimizer.step()
        if model.embedding_type == "bert":
            scheduler.step()
        epoch_loss += loss.item()
    print(epoch_index + 1, round(time() - time_now, 1), epoch_loss)

    acc, report = eval(model, test_dataloader)
    print("Accuracy:", acc)
    #print(report)
