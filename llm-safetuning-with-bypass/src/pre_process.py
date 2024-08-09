import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def remove_trailing_punctuation(text):
    """
    remove punctuation at end of the sentence.
    """
    # 处理空字符串
    if not text:
        return text
    # 去掉句末的空格
    text = text.rstrip()
    # 使用正则表达式去掉末尾所有标点符号
    return re.sub(r'[.,!?;:"\'“”]+$', '', text)


def add_punc(text):
    """
    add punctuation at end of the sentence.
    """
    if text[-1] != "." and text[-1] != "?":
        return text + "."
    else:
        return text


def remove_punctuation(content):
    """
    remove all punctuations in the sentence.
    """
    pattern = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5 ]")
    content = pattern.sub(' ', content)
    return content


def remove_whitespace(content):
    """
    remove multiple whitespace to single.
    """
    pattern = re.compile(r"[ ]{2,}")
    return pattern.sub(r' ', content)


def load_model(model_path, device):
    """
    load LLM from HuggingFace
    """
    model = AutoModelForCausalLM.from_pretrained(model_path, output_hidden_states=True, device_map="auto", torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def count_parameters(model):
    # 计算并打印总参数量
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


class HiddenDataset(Dataset):
    def __init__(self, dataframe):
        """
        Args:
            dataframe (pd.DataFrame): 包含数据的 DataFrame，第一列是标签，剩下的是提取的向量。
        """
        self.dataframe = dataframe
        self.labels = dataframe['label'].values
        self.prompts = dataframe['input'].values
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        prompt = self.prompts[idx]
        prompt = remove_punctuation(prompt)
        prompt = remove_whitespace(prompt)
        prompt = prompt.strip()
        return prompt, torch.tensor(label, dtype=torch.long)


class HiddenDataset_chat(Dataset):
    def __init__(self, dataframe, tokenizer):
        """
        Args:
            dataframe (pd.DataFrame): 包含数据的 DataFrame，第一列是标签，剩下的是提取的向量。
        """
        self.dataframe = dataframe
        self.labels = dataframe['label'].values
        self.prompts = dataframe['input'].values
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        prompt = self.prompts[idx]
        messages = [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        return text, torch.tensor(label, dtype=torch.long)


def func_extract_hidden_states(model, tokenizer, prompts, backbone_layer_num, device):
    """
    Given a LLM and specify the layer index, extract the hidden state (of all token positions).
    """
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(device)
    input_ids = inputs['input_ids']
    pad_masks = inputs["attention_mask"].bool()
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=pad_masks).hidden_states

    return outputs[backbone_layer_num].float(), pad_masks # torch.Size([batch, seq_len, 4096])

