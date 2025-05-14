export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer

e_layers=4
down_sampling_layers=1
down_sampling_window=2
learning_rate=0.01
d_model=32
d_ff=32
batch_size=16

python -u run.py \
  --use_gpu False \
  --gpu_type mps \
  --task_name short_term_forecast \
  --is_training 0 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Hourly' \
  --model_id m4_Hourly \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 128 \
  --d_model $d_model \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $learning_rate \
  --train_epochs 50 \
  --patience 20 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --loss 'SMAPE' 
