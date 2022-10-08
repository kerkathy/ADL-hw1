import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from re import X
from typing import Dict

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

from torch import nn 

# from seqeval.scheme import IOB2
# from seqeval.metrics import classification_report

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def main(args):

    # TODO: implement main function
    max_len=args.max_len
    hidden_size=args.hidden_size
    num_layers=args.num_layers
    dropout=args.dropout
    bidirectional=args.bidirectional
    lr=args.lr
    device=args.device
    num_epoch=args.num_epoch
    batch_size = args.batch_size
    trained_model_file = args.ckpt_dir / args.ckpt_name
    # trained_model_file = args.ckpt_dir / "model_slot.ckpt"
    
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, tag2idx, max_len)
        for split, split_data in data.items()
    }

    # TODO: create DataLoader for train / dev datasets
    train_dataloader = DataLoader(datasets[TRAIN], batch_size, collate_fn=datasets[TRAIN].collate_fn, shuffle=True)
    dev_dataloader = DataLoader(datasets[DEV], batch_size, collate_fn=datasets[DEV].collate_fn, shuffle=False)
    
    # embeddings: () -> tensor(num_samples, seq_len, embed_dim)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # TODO: init model and move model to target device(cpu / gpu)
    print(f"Using {device} device")
    
    model = SeqTagger(embeddings, hidden_size, num_layers, dropout, bidirectional, datasets[TRAIN].num_classes).to(device)
    print(model)
    
    # TODO: init optimizer
    ignore_index = datasets[DEV].num_classes
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index) # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
    optimizer = torch.optim.Adam(model.parameters(), lr)

    best_acc = 0.0
    train_len = len(datasets[TRAIN]) # num of training samples 
    val_len = len(datasets[DEV]) # num of val samples

    for epoch in range(num_epoch):
    # for epoch in epoch_pbar:
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        # for data in train_dataloader:
        for data in tqdm(train_dataloader):
            inputs, tags, ignore = data["tokens"], data["tags"], data["ignore"]
            inputs, tags, ignore = inputs.to(device), tags.to(device), ignore.to(device)
            # inputs: (batch_size=128, seq_len=30/18/21/...)
            # tags: (batch_size=128, seq_len=30)
            # ignore: (batch_size=128, seq_len=30)
            model.zero_grad() # empty the gradients to avoid accumulation
            
            outputs = model(inputs) # propagate forward
            # print("== Log & Softmax ==\noutput {}".format(outputs.size()))
            # outputs: tensor(batch_size=128, seq_len=26, class=9)

            optimizer.zero_grad()
            # Cross entropy input: (minibatch, C, d1, ..., dk); target(indices of label)
            # batch_loss = loss_fn(outputs.view(-1, outputs.shape[2]), tags)
            batch_loss = criterion(outputs.permute(0,2,1), tags)
            # batch_loss: scalar
            # outputs: tensor(seq_len=26, batch_size=128, class=9), tags: (batch_size=128, seq_len=30))

            _, train_pred = torch.max(outputs, 2) # get the index of the class with the highest probability, dim to reduce =2
            train_pred[ignore] = ignore_index

            # print(inputs[0])
            # print(tags[0])
            # print(ignore[0])
            # print(train_pred[0])
            # print("ignore_idx: {}".format(ignore_index))
            # exit()

            # train_pred: tensor(batch_size, seq_len)
            # print("train_pred: {}; mask:{}".format(train_pred.size(), mask.size()))


            batch_loss.backward() # backpropagation
            optimizer.step() # param update
        
            # print("train_pred {}; tags {}".format(train_pred.size(), tags.size()))
            # train_pred = (train_pred * mask) >= 0
            # train_acc += batch_acc
            # train_acc += (train_pred.cpu() == tags.cpu()).sum().item()
            train_acc += len(torch.where((train_pred==tags).all(dim=1))[0])
            train_loss += batch_loss.item()

        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval() # set the model to evaluation mode
        with torch.no_grad():
            for data in dev_dataloader:
                inputs, tags, ignore = data["tokens"], data["tags"], data["ignore"]
                inputs, tags, ignore = inputs.to(device), tags.to(device), ignore.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs.permute(0,2,1), tags)
                # batch_loss = loss_fn(outputs.view(-1, outputs.shape[2]), tags)

                _, val_pred = torch.max(outputs, 2)
                val_pred[ignore] = ignore_index
                # val_pred = val_pred * ignore # ignore paddings as 0

                # val_acc += (val_pred.cpu() == tags.cpu()).sum().item() # get the index of the class with the highest probability
                val_acc += len(torch.where((val_pred==tags).all(dim=1))[0])
                # val_acc += batch_acc
                val_loss += batch_loss.item()

            # seq_len = tags.size(1)
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc/train_len, train_loss, val_acc/val_len, val_loss
                # epoch + 1, num_epoch, train_acc/train_len/seq_len, train_loss/len(train_dataloader), val_acc/val_len/seq_len, val_loss/len(dev_dataloader)
            ))
            # classification_report(tags, val_pred, mode='strict', scheme=IOB2)

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), trained_model_file)
                print('saving model with acc {:.3f}'.format(best_acc))
                # print('saving model with acc {:.3f}'.format(best_acc/val_len/seq_len))

        pass

def loss_fn(outputs, labels):
    """
    Computes neg likelihood for a logged and softmaxed outputs vs. labels
    Masks out indices of negative ignore_idx. (<PAD> tokens)
    - outputs: (batch_size*batch_max_len, NUM_TAGS)
    - labels: (batch_size, batch_max_len)
    """
    # print("== Loss Fn ==\nOutputs: {} ; Labels: {}".format(outputs.size(), labels.size()))
    #reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.view(-1)  

    #mask out 'PAD' tokens
    mask = (labels > -1).float()

    #the number of tokens is the sum of elements in mask
    num_tokens = int(torch.sum(mask).item())

    #pick the values corresponding to labels and multiply by mask
    outputs = outputs[range(outputs.shape[0]), labels]*mask
    # print("outputs: {}".format(outputs.size()))
    # print("Pred: {} ; labels: {}".format(pred.size(), labels.size()))

    #cross entropy loss for all non 'PAD' tokens
    return -torch.sum(outputs)/num_tokens
    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )
    parser.add_argument(
        "--ckpt_name",
        type=Path,
        help="Name to save the model file.",
        required=True,
    )


    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)