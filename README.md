# Sample Code for Homework 1 ADL NTU

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl-hw1"
make
conda activate adl-hw1
# pip install -r requirements.txt
# Otherwise
pip install -r requirements.in
```

## Preprocessing
```shell
# Download necessary files
bash download.sh
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent detection
```shell
bash intent_cls.sh
```

## Slot tagging
```shell
bash slot_tag.sh
```

