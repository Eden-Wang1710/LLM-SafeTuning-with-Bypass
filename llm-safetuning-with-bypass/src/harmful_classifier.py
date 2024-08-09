from torch import nn
from src.pre_process import *


class HarmfulClassifier(nn.Module):
    """
    harmfulclassifier = HarmfulClassifier(tokenizer=tokenizer, backbone=model, safe_block=safe_by_pass, layer=layer_id)
    """
    def __init__(self, tokenizer, backbone, safe_block, layer=1):
        super().__init__()
        self.tokenizer = tokenizer
        self.backbone = backbone
        self.safe_block = safe_block      # safe by-pass (safe transformer +safe MLP)
        self.backbone_layer_num = layer

    # batch input
    def extract_hidden_states(self, prompts):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True).to(self.backbone.device)
        input_ids = inputs['input_ids']
        pad_masks = inputs["attention_mask"].bool()
        
        self.backbone.eval()
        with torch.no_grad():
            outputs = self.backbone(input_ids, attention_mask=pad_masks).hidden_states
    
        return outputs[self.backbone_layer_num].float(), pad_masks # torch.Size([batch, seq_len, 4096])

    # batch input
    def forward(self, prompts): 
        # prompts = [remove_whitespace(remove_punctuation(prompt)).strip() for prompt in prompts]
        prompts_ = []
        for p in prompts:
            messages = [
                # {"role": "system", "content": system_prompt},
                {"role": "user", "content": p}
            ]
        
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts_.append(text)
            
        self.backbone.eval()
        self.safe_block.eval()
        with torch.no_grad():
            hidden, pad_masks = self.extract_hidden_states(prompts_)
            output = self.safe_block(hidden, pad_mask=pad_masks, device=self.backbone.device)
            
            return output.argmax(-1)
