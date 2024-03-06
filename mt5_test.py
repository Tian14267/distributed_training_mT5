#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/4/23
# @Author : fffan
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration


model_path = "/data1/fffan/5_NLP/6_mT5/models/model_files_epoch_1_step_120000"
#model_path = "/data1/fffan/5_NLP/5_T5/models/t5_pretrain_model/t5-base"
tokenizer = MT5Tokenizer.from_pretrained(model_path)
model = MT5ForConditionalGeneration.from_pretrained(model_path)


# 修改colab笔记本设置为gpu，推理更快
device = torch.device('cuda')
model.to(device)


def preprocess(text):
	text = text.replace("\n", "\\n").replace("\t", "\\t")
	return text


def postprocess(text):
	return text.replace("\\n", "\n").replace("\\t", "\t")


def generate_prompt(instruction, input=None):
	if input:
		out = f"""用户:{instruction}。{input}### 小催:"""
		return out
	else:
		out = f"""用户:{instruction}### 小催:"""
		return out

def answer(text, sample=True, top_p=1, temperature=0.7):
	'''sample：是否抽样。生成任务，可以设置为True;
	top_p：0-1之间，生成的内容越多样'''
	text = preprocess(text)
	print(len(text))
	encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
	if not sample:
		out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=1000,
							 num_beams=1, length_penalty=0.6)
	else:
		out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=2000,
							 do_sample=True, top_p=top_p, temperature=temperature, no_repeat_ngram_size=3)
	out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
	return postprocess(out_text[0])


def rewrite_message(input):
	print("query message:", input)
	answer_message_list = []
	for each in range(4):
		answer_message_list.append("方案{0}：".format(each) + answer_message(input))

	return "\n\n".join(answer_message_list)


def answer_message(input):
	input_format = input.replace("\n", "。")
	input_text = input_format + "\n："
	output_text = answer(input_text)
	print(output_text)
	return f"{output_text}"



if __name__ == '__main__':
	#answer_message("人工客服上班时间几点？")
	print("####  请输入你的要求：")
	input_t = input()
	while input_t:
		input_format = input_t.replace("\n", "。")
		#input_text = generate_prompt(input_format)
		res = answer(input_format)
		print(f"回答：{res}")
		print("\n-------------------------------\n")
		print("####  请输入你的要求：")
		input_t = input()
