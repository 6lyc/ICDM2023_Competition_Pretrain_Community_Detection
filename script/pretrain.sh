# pretrain our model on ogbn_arxiv
nohup python -u train.py --mode "pretrain" --exp 1 > ./logs/log1_pretrain_ogbn_arxiv_exp1pretraining.log 2>&1 &

wait

# fine tune our model on iicdm2023_session1_test
nohup python -u train.py --mode "ft" --exp 1 > ./logs/log1_test_icdm_data_exp1pretrain.log 2>&1 &