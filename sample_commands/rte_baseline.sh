# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Dina EL ZEIN <dina.el-zein@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only


task_name="rte"
exp="baseline"
model="bert_large"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/log}"

log_dir="${LOG_ROOT}/${model}/${task_name}/${exp}"


for seed in `seq 1 5`; do
    seed_log_dir="${log_dir}/${seed}" 
    mkdir -p "${seed_log_dir}"
    touch "${seed_log_dir}/output.out"
    touch "${seed_log_dir}/error.err"

    python main.py \
        --model_name_or_path google-bert/bert-large-uncased \
        --task_name ${task_name} \
        --exp_type '' \
        --max_seq_length 128 \
        --eval_batch_size 16 \
        --train_batch_size 32 \
        --learning_rate 2e-5 \
        --max_epochs 10 \
        --cache_dir cache \
        --seed $seed \
        --output_dir output/${model}/${task_name}/${exp}/${seed} \
        --log_dir log/${model}/${task_name}/${exp}/${seed} \
        --output_file results/${model}/${task_name}/${exp}.csv \
        --dropout 0 \
        --weight_decay 0 \
        > "${seed_log_dir}/output.out" \
        2> "${seed_log_dir}/error.err"
done 

