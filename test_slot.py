import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

import csv


TEST = "test"

def main(args):

    max_len=args.max_len
    hidden_size=args.hidden_size
    num_layers=args.num_layers
    dropout=args.dropout
    bidirectional=args.bidirectional
    device=args.device
    batch_size = args.batch_size
    ckpt_path=args.ckpt_path
    
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, max_len)

    # TODO: crecate DataLoader for test dataset
    test_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, shuffle=False)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTagger(embeddings, hidden_size, num_layers, dropout, bidirectional, dataset.num_classes).to(device)

    ckpt = torch.load(ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)

    # TODO: predict dataset
    predict = []
    all_ids = []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs = data["tokens"]
            ids = data["id"]
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, test_pred = torch.max(outputs, 2) # get the index of the class with the highest probability

            test_pred = test_pred.t()
            # test_pred: (batch_size, seq_len)

            for sentence in test_pred.cpu().numpy():
                predict.append([dataset.idx2label(word) for word in sentence if word != -1])
            for id in ids:
                all_ids.append(id)

    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        f.write('id,intent\n')
        for id, y in zip(all_ids, predict):
            # TODO: id2label
            f.write('{},'.format(id))
            writer.writerow(y)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    # parser.add_argument(
    #     "--ckpt_dir",
    #     type=Path,
    #     help="Directory to save the model file.",
    #     default="./ckpt/slot/",
    # )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, help="Path to save the predicted file", required=True, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)