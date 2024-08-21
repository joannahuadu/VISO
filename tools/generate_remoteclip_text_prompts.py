import json
import argparse
import numpy as np
import open_clip

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='ViT-B-32')
    parser.add_argument(
        '--pretrained',
        type=str,
        default='weights/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-B-32.pt')
    parser.add_argument('--text',
                        type=str,
                        default='data/captions/coco_class_captions.json')
    parser.add_argument('--out', type=str, default='output.npy')

    args = parser.parse_args()
    
    tokenizer = open_clip.get_tokenizer(args.model)
    model = open_clip.create_model(args.model, args.pretrained)

    with open(args.text) as f:
        data = json.load(f)
    texts = [x[0] for x in data]
    device = 'cuda:0'
    model.to(device)
    texts = tokenizer(texts=texts)
    texts = texts.to(device)
    txt_feats = model.encode_text(texts, normalize=True)
    txt_feats /= txt_feats.norm(dim=-1, keepdim=True)
    txt_feats = txt_feats.reshape(-1, txt_feats.shape[-1])
    
    np.save(args.out, txt_feats.cpu().data.numpy())
