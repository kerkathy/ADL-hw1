# Homework 1 ADL NTU

## Use the Trained Model to Predict
### Environment
Assume that you already have the environment as the one stated in the sample code provided by TAs.
If not, please run
```shell
make
```

### Preprocessing
This will download `/ckpt` and `/cache` folder which contains the best model checkpoints and pretrained embeddings respectively.  
```shell
# Download necessary files
bash download.sh
```

### Intent classification
This will run the test for intent classification given the test data provided in prompt. A `pred.intent.csv` should be generated.
```shell
bash intent_cls.sh /path/to/test.json /path/to/pred.csv
# bash intent_cls.sh data/intent/test.json pred.intent.csv
```

### Slot tagging
This will run the test for slot tagging given the test data provided in prompt. A `pred.intent.csv` should be generated.
```shell
bash slot_tag.sh /path/to/test.json /path/to/pred.csv
# bash slot_tag.sh data/slot/test.json pred.slot.csv
```

## Reproduce the Trained Checkpoint
### Preprocessing
This will download `/ckpt` and `/cache` folder which contains the best model checkpoints and pretrained embeddings respectively.  
```shell
# Download necessary files
bash download.sh
```

### Intent classification
This will first train a intent classification model using `data/intent/train.json` and `data/intent/eval.json`.
Then, run the test for intent classification using `data/intent/test.json`. A `pred.intent.csv` should be generated.
```shell
python3 train_intent.py --ckpt_name intent_best.ckpt --num_epoch 80 --batch_size 256 --dropout 0.6
python3 test_intent.py --ckpt_path intent_best.ckpt --pred_flie pred.intent.csv --test_file data/intent/test.json 
```

### Slot tagging
This will first train a slot tagging model using `data/slot/train.json` and `data/slot/eval.json`.
Then, run the test for slot tagging using `data/intent/test.json`. A `pred.slot.csv` should be generated.
```shell
python3 train_slot.py --ckpt_name slot_best.ckpt --num_epoch 60 --dropout 0.6
python3 test_slot.py --ckpt_path slot_best.ckpt --pred_flie pred.slot.csv --test_file data/slot/test.json 
```
