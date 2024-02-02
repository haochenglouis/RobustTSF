export CUDA_VISIBLE_DEVICES=4
## electricity
cd ..



python -u run.py \
 --is_training 1 \
 --data_path electricity.csv \
 --task_id ECL \
 --model Informer \
 --data custom \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len 1 \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --des 'Exp' \
 --itr 3 \
 --noise_ratio 0.1 \
 --noise_type const


python -u run.py \
 --is_training 1 \
 --data_path electricity.csv \
 --task_id ECL \
 --model Informer \
 --data custom \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len 1 \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --des 'Exp' \
 --itr 3 \
 --noise_ratio 0.3 \
 --noise_type const


python -u run.py \
 --is_training 1 \
 --data_path electricity.csv \
 --task_id ECL \
 --model Informer \
 --data custom \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len 1 \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --des 'Exp' \
 --itr 3 \
 --noise_ratio 0.1 \
 --noise_type missing


python -u run.py \
 --is_training 1 \
 --data_path electricity.csv \
 --task_id ECL \
 --model Informer \
 --data custom \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len 1 \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --des 'Exp' \
 --itr 3 \
 --noise_ratio 0.3 \
 --noise_type missing



python -u run.py \
 --is_training 1 \
 --data_path electricity.csv \
 --task_id ECL \
 --model Informer \
 --data custom \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len 1 \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --des 'Exp' \
 --itr 3 \
 --noise_ratio 0.1 \
 --noise_type gaussian


python -u run.py \
 --is_training 1 \
 --data_path electricity.csv \
 --task_id ECL \
 --model Informer \
 --data custom \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len 1 \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --des 'Exp' \
 --itr 3 \
 --noise_ratio 0.3 \
 --noise_type gaussian








