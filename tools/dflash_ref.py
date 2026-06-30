import torch, numpy as np, torch.nn.functional as F
from safetensors import safe_open
torch.set_grad_enabled(False)
H,NH,NKV,HD,eps,theta=2816,32,8,128,1e-6,1e6
DR='/models/gemma-4-26B-A4B-DFlash/model.safetensors'
W={};
with safe_open(DR,'pt') as f:
    for k in f.keys(): W[k]=f.get_tensor(k).float()
# target embed rows for the block qids only
meta=open('/tmp/dbg_meta.txt').read().strip().split('\n')
C,BLK,k=map(int,meta[0].split())
ctx_pos=torch.tensor(list(map(int,meta[1].split())))
qids=list(map(int,meta[2].split()))
qpos=torch.tensor(list(map(int,meta[3].split())))
cuda_prop=list(map(int,meta[4].split()))
taps=torch.from_numpy(np.fromfile('/tmp/dbg_taps.bin',dtype=np.float32)).view(C,6,H)
cuda_hfin=torch.from_numpy(np.fromfile('/tmp/dbg_hfin.bin',dtype=np.float32)).view(BLK,H)
with safe_open('/models/gemma-4-26B-A4B-it-NVFP4/model.safetensors','pt') as f:
    emb_t=f.get_slice('model.language_model.embed_tokens.weight')
    embrows=torch.stack([emb_t[q].float() for q in qids])  # [BLK,H] bf16->float

def rms(x,w): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+eps)*w
def rope(x,pos):  # x [seq,nh,HD], neox rotate-half
    inv=1.0/(theta**(torch.arange(0,HD,2).float()/HD)); ang=pos[:,None].float()*inv[None,:]
    cos=torch.cos(ang)[:,None,:]; sin=torch.sin(ang)[:,None,:]
    x1,x2=x[...,:HD//2],x[...,HD//2:]
    return torch.cat([x1*cos-x2*sin, x2*cos+x1*sin],-1)

import sys
ESC=53.06599664 if len(sys.argv)<2 or sys.argv[1]!='unscaled' else 1.0
# ---- context K/V per layer ----
ctx_states = taps.reshape(C,6*H) @ W['fc.weight'].T          # fc
ctx_n = rms(ctx_states, W['hidden_norm.weight'])             # hidden_norm
Kc=[];Vc=[]
for l in range(5):
    kc=(ctx_n@W[f'layers.{l}.self_attn.k_proj.weight'].T).view(C,NKV,HD)
    vc=(ctx_n@W[f'layers.{l}.self_attn.v_proj.weight'].T).view(C,NKV,HD)
    kc=rms(kc,W[f'layers.{l}.self_attn.k_norm.weight']); kc=rope(kc,ctx_pos)
    Kc.append(kc);Vc.append(vc)
# ---- block forward ----
h=embrows*ESC; residual=None
for l in range(5):
    if residual is None: residual=h; hn=rms(h,W[f'layers.{l}.input_layernorm.weight'])
    else: residual=residual+h; hn=rms(residual,W[f'layers.{l}.input_layernorm.weight'])
    q=(hn@W[f'layers.{l}.self_attn.q_proj.weight'].T).view(BLK,NH,HD)
    kb=(hn@W[f'layers.{l}.self_attn.k_proj.weight'].T).view(BLK,NKV,HD)
    vb=(hn@W[f'layers.{l}.self_attn.v_proj.weight'].T).view(BLK,NKV,HD)
    q=rms(q,W[f'layers.{l}.self_attn.q_norm.weight']); kb=rms(kb,W[f'layers.{l}.self_attn.k_norm.weight'])
    q=rope(q,qpos); kb=rope(kb,qpos)
    Kall=torch.cat([Kc[l],kb],0); Vall=torch.cat([Vc[l],vb],0)  # [C+BLK,NKV,HD]
    causal = (l<4)
    out=torch.zeros(BLK,NH,HD)
    for hh in range(NH):
        kvh=hh//(NH//NKV)
        sc=(q[:,hh,:]@Kall[:,kvh,:].T)*(HD**-0.5)  # [BLK, C+BLK]
        for i in range(BLK):
            # context always; block keys: causal -> j<=i, full -> all
            mask=torch.zeros(C+BLK,dtype=torch.bool); mask[:C]=True
            if causal: mask[C:C+i+1]=True
            else: mask[C:]=True
            s=sc[i].masked_fill(~mask,-1e30); w=torch.softmax(s,-1)
            out[i,hh,:]=w@Vall[:,kvh,:]
    ao=out.reshape(BLK,NH*HD)@W[f'layers.{l}.self_attn.o_proj.weight'].T
    residual=residual+ao; hn2=rms(residual,W[f'layers.{l}.post_attention_layernorm.weight'])
    g=hn2@W[f'layers.{l}.mlp.gate_proj.weight'].T; u=hn2@W[f'layers.{l}.mlp.up_proj.weight'].T
    h=(F.silu(g)*u)@W[f'layers.{l}.mlp.down_proj.weight'].T
residual=residual+h; hfin=rms(residual,W['norm.weight'])
# ---- proposals ----
logits=hfin[1:1+k]@embrows.new_tensor([]).new_zeros(0) if False else None
# lm_head = target embed (need full table for argmax) -> load lazily
with safe_open('/models/gemma-4-26B-A4B-it-NVFP4/model.safetensors','pt') as f:
    LM=f.get_tensor('model.language_model.embed_tokens.weight').float()
lg=hfin[1:1+k]@LM.T  # [k,VOCAB]
ref_prop=lg.argmax(-1).tolist()
print(f"ESC={ESC}")
print("hfin diff (mine vs cuda): max",(hfin-cuda_hfin).abs().max().item(),"mean",(hfin-cuda_hfin).abs().mean().item(),"cuda_norm",cuda_hfin.abs().mean().item())
print("ref proposed: ",ref_prop)
print("cuda proposed:",cuda_prop)
print("match:",sum(int(a==b) for a,b in zip(ref_prop,cuda_prop)),"/",k)
