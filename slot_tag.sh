# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
# run: bash ./slot_tag.sh /path/to/test.json /path/to/pred
python3 test_slot.py --test_file "${1}" --pred_file "${2}" --ckpt_path ckpt/slot/best.ckpt 
