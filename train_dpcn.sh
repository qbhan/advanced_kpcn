python train.py \
  --mode 'dpcn' \
  --input_channels 28 \
  --hidden_channels 100 \
  --num_layer 9 \
  --eps 0.00316 \
  --do_val \
  --lr 1e-4 \
  --epochs 20 \
  --loss 'L1'