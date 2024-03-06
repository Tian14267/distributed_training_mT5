#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: fffan
@Time: 2023-06-19
@comment:
    安装 NCCL
    安装 horovod：0.27.0
        安装方法：HOROVOD_GPU_OPERATIONS=NCCL pip install horovod
    使用：CUDA_VISIBLE_DEVICES="1,2,3" horovodrun -np 3 python mT5_train.py
    https://github.com/horovod/horovod/blob/master/docs/pytorch.rst
"""
import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from torch import cuda
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from loguru import logger
from data import prepare_data,prepare_data_new
from torch_optimizer import Adafactor
from dialogdataset import DialogDataSet
# Importing the MT5 modules from huggingface/transformers
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import horovod.torch as hvd
from torchinfo import summary

hvd.init()
logger.info("hvd.local_rank:{} ".format(hvd.local_rank()))
logger.info("hvd.rank:{} ".format(hvd.rank()))
logger.info("hvd.local_size:{} ".format(hvd.local_size()))
logger.info("hvd.size:{} ".format(hvd.size()))

torch.cuda.set_device(hvd.local_rank())
#os.environ["CUDA_VISIBLE_DEVICES"] = str(hvd.local_rank())
device = 'cpu'




def train(epoch, tokenizer, model, device, loader, optimizer, accumulation_step, output_dir):
    """
    用于训练的方法
    """

    model.train()
    time1 = time.time()
    logger.info(f"###############################  train all step in epoch {epoch} is : {len(loader)} ")
    for step, data in enumerate(tqdm(loader,desc=f'Train epoch {epoch}')):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()  # target, from start to end(except end of token, <EOS>). e.g. "你好吗？"
        lm_labels = y[:, 1:].clone().detach()  # target, for second to end.e.g."好吗？<EOS>"
        lm_labels[y[:,
                  1:] == tokenizer.pad_token_id] = -100  # releted to pad_token and loss. for detail, check here: https://github.com/Shivanandroy/T5-Finetuning-PyTorch/issues/3
        ids = data["source_ids"].to(device, dtype=torch.long)  # input. e.g. "how are you?"
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]
        loss = loss.mean()
        loss = loss / accumulation_step
        loss.backward()

        if epoch == 0 and step == 0 and hvd.rank() == 0:   ###  只在训练开头打印参数信息，其他时候不打印
            summary(model=model, input_ids=ids,attention_mask=mask,decoder_input_ids=y_ids,labels=lm_labels, device=device)


        # training_logger.add_row(str(epoch), str(_), str(loss))
        # console.logger.info(training_logger)
        if (step + 1) % accumulation_step == 0:
            optimizer.step()
            optimizer.zero_grad()
            if (step + 1) % (accumulation_step * 3) == 0:
                time2 = time.time()
                logger.info(
                    "step: " + str(step) + " epoch:" + str(epoch) + "-loss:" + str(loss) + "; iter time spent:" + str(
                        float(time2 - time1)))
                time1 = time.time()

        if step != 0 and step % 30000 ==0:
            #  save model for step
            if hvd.rank() == 0:
                logger.info(f"[Saving Model]...\n")
                path = os.path.join(output_dir, "model_files" + "_epoch_{}_step_{}".format(epoch,step))
                model.save_pretrained(path)
                tokenizer.save_pretrained(path)


def validate(tokenizer, model, loader, max_length):
    """
    用于验证的方法：输入用于验证的数据，返回模型预测的结果和正确的标签

    """
    model.eval()
    predictions = []
    actuals = []
    source_list = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)
            source_text = data["source_text"]

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=max_length,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                     generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
            logger.info("source is: {} \npreds is: {} \ntarget is: {}".format(source_text, preds, target))
            if _ % 1000 == 0:
                logger.info(f'Completed {_}')
            source_list.extend(source_text)
            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals, source_list


# 训练类：整合数据集类、训练方法、验证方法，加载数据进行训练并验证训练过程的效果
def MT5Trainer(args):
    """
    MT5 trainer
    """
    logger.info("trainer begin")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    #logger.info(f"""[Model]: Loading {args.model_name_or_path}...\n""")
    #logger.info("gpu number!: {}".format(torch.cuda.device_count()))

    #
    tokenizer = MT5Tokenizer.from_pretrained(args.model_name_or_path)

    #
    model = MT5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.to(device)
    # logging
    logger.info(f"[Data]: Reading data...\n")

    #
    train_data_list, val_data_list = prepare_data_new(args.data_file,
                                                      args.max_source_text_length,
                                                      args.max_target_text_length)

    #
    train_dataset = DialogDataSet(
        train_data_list,
        tokenizer,
        args.max_source_text_length,
        args.train_batch_size * args.accumulation_step * torch.cuda.device_count()
        # trick，手动丢弃多余数据
    )

    logger.info("length of training dataset is: {}".format(len(train_dataset)))

    val_dataset = DialogDataSet(
        val_data_list,
        tokenizer,
        args.max_source_text_length
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(),
                                                                    rank=hvd.rank(), shuffle=True)
    #
    train_params = {
        "batch_size": args.train_batch_size,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": args.valid_batch_size,
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation.
    training_loader = DataLoader(train_dataset, batch_size=train_params["batch_size"], shuffle=False,
                                 sampler=train_sampler)
    val_loader = DataLoader(val_dataset, **val_params)

    # mT5训练optimizer建议使用Adafactor，见论文原文。
    optimizer = Adafactor(
        params=model.parameters(), lr=args.learning_rate / hvd.size()
    )
    optimizer = hvd.DistributedOptimizer(optimizer, backward_passes_per_step=args.accumulation_step)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    # Training loop
    logger.info(f"[Initiating Fine Tuning]...\n")
    logger.info("the length of dataloader is: {}".format(len(training_loader)))

    for epoch in range(args.train_epochs):
        # 1) train for one epoch
        train(epoch, tokenizer, model, device, training_loader, optimizer, args.accumulation_step,output_dir)

        # 2) save model for each epoch
        if hvd.rank() == 0:
            logger.info(f"[Saving Model]...\n")
            path = os.path.join(output_dir, "model_files" + "_epoch_{}".format(epoch))
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)

            # 3) evaluating test dataset
            logger.info(f"[Initiating Validation]...\n")
            with torch.no_grad():
                if epoch != args.train_epochs - 1:
                    continue
                predictions, actuals, source = validate(tokenizer, model, val_loader,
                                                        args.max_target_text_length)
                predict_path = output_dir + "epoch_{}".format(epoch) + "_predictions.csv"
                final_df = pd.DataFrame({"source_text": source, "Generated Text": predictions, "Actual Text": actuals})
                final_df.to_csv(predict_path, index=False, sep="\t")

    logger.info(f"[Validation Completed.]\n")
    logger.info(
        f"""[Model] Model saved @ {os.path.join(args.output_dir, "model_files")}\n"""
    )
    logger.info(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(args.output_dir, 'predictions.csv')}\n"""
    )
    logger.info(f"""[Logs] Logs saved @ {os.path.join(args.output_dir, 'logs.txt')}\n""")


def main():
    parser = argparse.ArgumentParser(description='Args of mt5')

    # Model Args
    parser.add_argument('--model_name_or_path', default="/data1/fffan/5_NLP/5_T5/models/mt5_pretrain_model/mt5-xl",
                        type=str)

    parser.add_argument('--train_batch_size', default=1, type=int)
    parser.add_argument('--valid_batch_size', default=1, type=int)
    parser.add_argument('--train_epochs', default=10, type=int)
    parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--max_source_text_length', default=128, type=int)
    parser.add_argument('--max_target_text_length', default=128, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--accumulation_step', default=2, type=int)
    parser.add_argument('--data_file', default="/data1/fffan/5_NLP/6_mT5/data/cuishou_train_file_vicuna.json", type=str)
    parser.add_argument('--output_dir', default="./outputs/mt5_finetune", type=str)

    args = parser.parse_args()

    MT5Trainer(args)



if __name__ == "__main__":
    main()
