import json
import argparse
import numpy as np
from transformers import (AutoTokenizer, CLIPTextModelWithProjection)
import torch

def load_model_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
    model_state_dict = checkpoint.get('state_dict', checkpoint)
    clip_model_state_dict = {k[26:]: v for k, v in model_state_dict.items() if 'text_model' in k}

    model.load_state_dict(clip_model_state_dict, strict=False)

    return model
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='openai/clip-vit-base-patch32')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None)
    parser.add_argument('--text',
                        type=str,
                        default='data/captions/coco_class_captions.json')
    parser.add_argument('--out', type=str, default='output.npy')

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = CLIPTextModelWithProjection.from_pretrained(args.model)
    if args.checkpoint is not None:
        load_model_checkpoint(model, args.checkpoint)
    with open(args.text) as f:
        data = json.load(f)
    prompt = "Detect the {}"
    # prompt = "{}"
    texts = [prompt.format(x[0]) for x in data]
    device = 'cuda:0'
    model.to(device)
    texts = tokenizer(text=texts, return_tensors='pt', padding=True)
    texts = texts.to(device)
    text_outputs = model(**texts)
    txt_feats = text_outputs.text_embeds
    txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
    txt_feats = txt_feats.reshape(-1, txt_feats.shape[-1])

    np.save(args.out, txt_feats.cpu().data.numpy())
