TOT_CUDA="3,4"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}

#DATA_PATH="../data/0.5m_concat_cuishou.json" #"../dataset/instruction/guanaco_non_chat_mini_52K-utf8.json" #"./sample/merge_sample.json"
DATA_PATH="../data/cuishou_train_file_vicuna.json" #"../dataset/instruction/guanaco_non_chat_mini_52K-utf8.json" #"./sample/merge_sample.json"
OUTPUT_PATH="./output/05m_concat_cuishou_mutilgpu"
MODEL_PATH="/data1/fffan/5_NLP/5_T5/models/mt5_pretrain_model/mt5-base"
#lora_checkpoint="./lora-Vicuna/checkpoint-11600"
TEST_SIZE=1

TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES="3,4" python -m torch.distributed.launch --nproc_per_node=$CUDA_NUM mt5_train_0512.py \
--data_path $DATA_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH \
--eval_steps 200 \
--save_steps 200 \
--test_size $TEST_SIZE
