// forward.cu — Gemma-4-26B-A4B NVFP4 single-sequence forward (W4A4), correctness-first.
// Reuses verified kernels: nvfp4_gemm, nvfp4_quantize_activations, rmsnorm, rope, sdpa, gelu, softcap.
// Produces next-token logits for a token-id sequence (prefill); gate compares to the reference server.
#include "safetensors.h"
#include "fp4_gemm.h"
#include "nvfp4_quant.h"
#include "elementwise.h"
#include "attention.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>

#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

// ---- Gemma-4-26B-A4B config (verified) ----
static const int H=2816, NLAYER=30, NHEAD=16, HD_S=256, NKV_S=8, HD_F=512, NKV_F=2;
static const int VOCAB=262144, NEXP=128, TOPK=8, MOE_INT=704, MLP_INT=2112, SWIN=1024;
static const float EPS=1e-6f, SOFTCAP=30.0f;
static const float EMB_SCALE=53.06599664f;             // sqrt(2816)
static inline bool is_full(int L){ return L==5||L==11||L==17||L==23||L==29; }

// ---- helper kernels ----
__device__ __forceinline__ float bf2f(uint16_t h){ unsigned u=(unsigned)h<<16; float f; memcpy(&f,&u,4); return f; }
__global__ void k_transpose(float* out,const float* in,int M,int N,int ld){ // in col-major (ld) -> out row-major[M,N]
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=M*N)return; int m=i/N,n=i%N; out[(size_t)m*N+n]=in[(size_t)m+(size_t)n*ld]; }
__global__ void k_embed(float* out,const uint16_t* emb,const int* ids,int seq,int H,float scale){
    int t=blockIdx.x,i=threadIdx.x; for(int j=i;j<H;j+=blockDim.x) out[(size_t)t*H+j]=bf2f(emb[(size_t)ids[t]*H+j])*scale; }
__global__ void k_mul(float* a,const float* b,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n)a[i]*=b[i]; }
__global__ void k_scale_rows(float* x,const float* s,int rows,int dim){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<rows*dim)x[i]*=s[i/dim]; }
__global__ void k_gather(float* out,const float* in,const int* idx,int n,int dim){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n*dim)out[i]=in[(size_t)idx[i/dim]*dim + i%dim]; }
__global__ void k_scatter_add(float* out,const float* in,const int* idx,const float* w,int n,int dim){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n*dim)return; int r=i/dim; atomicAdd(&out[(size_t)idx[r]*dim + i%dim], in[i]*w[r]); }
// lm_head for the LAST token: logits[v]=softcap(sum_h hlast[h]*bf16(emb[v,h])); one block per vocab
__global__ void k_lmhead(float* logits,const float* hlast,const uint16_t* emb,int H,int V,float cap){
    int v=blockIdx.x; if(v>=V)return; __shared__ float red[256];
    float acc=0; for(int h=threadIdx.x;h<H;h+=blockDim.x) acc+=hlast[h]*bf2f(emb[(size_t)v*H+h]);
    red[threadIdx.x]=acc; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s)red[threadIdx.x]+=red[threadIdx.x+s]; __syncthreads(); }
    if(threadIdx.x==0){ float x=red[0]; logits[v]=cap*tanhf(x/cap); } }

// ---- Model ----
struct Model {
    st::SafeTensors* st; uint8_t* draw; const uint8_t* h0;
    cublasLtHandle_t lt; void* ws; size_t wsb=64u<<20;
    std::unordered_map<std::string,uint8_t*> swz;   // swizzled scales
    std::unordered_map<std::string,uint8_t*> wpk;   // aligned packed-weight copies (cublasLt needs 16B align)
    std::unordered_map<std::string,float*> norms;   // fp32 norm weights
    Model(const std::string& path){
        st=new st::SafeTensors(path); h0=st->dataStart();
        CU(cudaMalloc(&draw, st->dataBytes()));
        CU(cudaMemcpy(draw, h0, st->dataBytes(), cudaMemcpyHostToDevice));
        cublasLtCreate(&lt); CU(cudaMalloc(&ws,wsb));
        printf("uploaded %.1f GB weights to device\n", st->dataBytes()/1e9);
    }
    const st::Tensor& T(const std::string& n){ return st->get(n); }
    bool has(const std::string& n){ return st->has(n); }
    template<class P> P dptr(const std::string& n){ return (P)(draw + (st->get(n).data - h0)); }
    float scalarF32(const std::string& n){ float v; memcpy(&v, st->get(n).data, 4); return v; }
    float scalarBf16(const std::string& n){ uint16_t b; memcpy(&b, st->get(n).data,2); unsigned u=(unsigned)b<<16; float f; memcpy(&f,&u,4); return f; }
    // aligned packed-weight device copy (cached) — cublasLt FP4 requires aligned operands
    uint8_t* wpacked(const std::string& wname){
        auto it=wpk.find(wname); if(it!=wpk.end())return it->second;
        const auto& t=T(wname+".weight_packed"); uint8_t* d; CU(cudaMalloc(&d,t.nbytes));
        CU(cudaMemcpy(d,t.data,t.nbytes,cudaMemcpyHostToDevice)); wpk[wname]=d; return d;
    }
    // swizzled scale device ptr (cached). weight_scale shape [N, K/16]
    uint8_t* wscale(const std::string& wname){
        auto it=swz.find(wname); if(it!=swz.end())return it->second;
        const auto& ts=T(wname+".weight_scale"); int N=ts.shape[0], K=ts.shape[1]*16;
        size_t bytes=nvfp4_scale_buffer_bytes(N,K);
        std::vector<uint8_t> host(bytes); nvfp4_swizzle_scales(ts.data, host.data(), N, K);
        uint8_t* d; CU(cudaMalloc(&d,bytes)); CU(cudaMemcpy(d,host.data(),bytes,cudaMemcpyHostToDevice));
        swz[wname]=d; return d;
    }
    // fp32 norm weight (cached) from bf16
    float* norm(const std::string& n){
        auto it=norms.find(n); if(it!=norms.end())return it->second;
        const auto& t=T(n); int d=t.numel(); std::vector<float> hf(d);
        const uint16_t* b=(const uint16_t*)t.data;
        for(int i=0;i<d;++i){ unsigned u=(unsigned)b[i]<<16; memcpy(&hf[i],&u,4);}
        float* dp; CU(cudaMalloc(&dp,d*4)); CU(cudaMemcpy(dp,hf.data(),d*4,cudaMemcpyHostToDevice));
        norms[n]=dp; return dp;
    }
};

// scratch (reused)
struct Scratch { float* in_pad; uint8_t* xp; uint8_t* xs; float* dcol; size_t cap_rows=512, cap_k=8192, cap_n=8192;
    Scratch(){ CU(cudaMalloc(&in_pad, cap_rows*cap_k*4)); CU(cudaMalloc(&xp, cap_rows*cap_k/2));
        CU(cudaMalloc(&xs, nvfp4_scale_buffer_bytes(cap_rows,cap_k))); CU(cudaMalloc(&dcol, cap_rows*cap_n*4)); } };
static Scratch* SC;

// W4A4 Linear: out_row[M,N] = (in_row[M,K] @ W[N,K]^T) using stored global scales. Pads M>=128.
static void linear(Model& m, float* out_row, const float* in_row, const std::string& prefix, int M, int N, int K){
    int Mp = M<128?128:M;
    CU(cudaMemset(SC->in_pad, 0, (size_t)Mp*K*4));
    CU(cudaMemcpy(SC->in_pad, in_row, (size_t)M*K*4, cudaMemcpyDeviceToDevice));
    float ig = m.scalarF32(prefix+".input_global_scale");
    float wg = m.scalarF32(prefix+".weight_global_scale");
    CU(cudaMemset(SC->xs, 0, nvfp4_scale_buffer_bytes(Mp,K)));
    nvfp4_quantize_activations(SC->in_pad, SC->xp, SC->xs, Mp, K, ig, 0);
    uint8_t* Wp = m.wpacked(prefix);
    uint8_t* Ws = m.wscale(prefix);
    int rc = nvfp4_gemm(SC->dcol, SC->xp, SC->xs, 1.0f/ig, Wp, Ws, 1.0f/wg, Mp, N, K, m.lt, m.ws, m.wsb, 0);
    if(rc){ fprintf(stderr,"linear %s rc=%d\n",prefix.c_str(),rc); exit(1);}
    CU(cudaDeviceSynchronize());
    k_transpose<<<(M*N+255)/256,256>>>(out_row, SC->dcol, M, N, Mp); // first M rows; ld=Mp
}

// build rope cos/sin tables [seq, half] on host -> device
static void rope_tables(int seq,int head_dim,double theta,int rope_angles,float** dcos,float** dsin){
    int half=head_dim/2; std::vector<float> c((size_t)seq*half), s((size_t)seq*half);
    for(int p=0;p<seq;++p) for(int j=0;j<half;++j){
        double inv = (j<rope_angles)? 1.0/pow(theta, (2.0*j)/head_dim) : 0.0;
        double a=p*inv; c[(size_t)p*half+j]=cosf(a); s[(size_t)p*half+j]=sinf(a);
    }
    CU(cudaMalloc(dcos,c.size()*4)); CU(cudaMalloc(dsin,s.size()*4));
    CU(cudaMemcpy(*dcos,c.data(),c.size()*4,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(*dsin,s.data(),s.size()*4,cudaMemcpyHostToDevice));
}

// ---- attention block: h_io[seq,H] updated in place (residual handled by caller? no, here) ----
static void attention(Model& m, float* h, int seq, int L){
    std::string P="model.language_model.layers."+std::to_string(L)+".";
    int hd = is_full(L)?HD_F:HD_S, nkv = is_full(L)?NKV_F:NKV_S;
    int qd=NHEAD*hd, kd=nkv*hd;
    float *hn,*q,*k,*v,*ao;
    CU(cudaMalloc(&hn,(size_t)seq*H*4)); CU(cudaMalloc(&q,(size_t)seq*qd*4));
    CU(cudaMalloc(&k,(size_t)seq*kd*4)); CU(cudaMalloc(&v,(size_t)seq*kd*4)); CU(cudaMalloc(&ao,(size_t)seq*qd*4));
    rmsnorm(hn, h, m.dptr<const uint16_t*>(P+"input_layernorm.weight"), seq, H, EPS, 0);
    linear(m, q, hn, P+"self_attn.q_proj", seq, qd, H);
    linear(m, k, hn, P+"self_attn.k_proj", seq, kd, H);
    if(is_full(L)) CU(cudaMemcpy(v, k, (size_t)seq*kd*4, cudaMemcpyDeviceToDevice)); // k_eq_v: v = raw k_proj
    else linear(m, v, hn, P+"self_attn.v_proj", seq, kd, H);
    // per-head q/k norm (weight [hd]); v_norm no weight
    rmsnorm(q, q, m.dptr<const uint16_t*>(P+"self_attn.q_norm.weight"), seq*NHEAD, hd, EPS, 0);
    rmsnorm(k, k, m.dptr<const uint16_t*>(P+"self_attn.k_norm.weight"), seq*nkv, hd, EPS, 0);
    rmsnorm(v, v, (const uint16_t*)nullptr, seq*nkv, hd, EPS, 0); // v_norm with_scale=false
    // rope
    float *dc,*ds; double theta=is_full(L)?1e6:1e4; int rang=is_full(L)?64:128;
    rope_tables(seq, hd, theta, rang, &dc, &ds);
    rope_rotate_half(q, dc, ds, seq, NHEAD, hd, hd, 0);
    rope_rotate_half(k, dc, ds, seq, nkv,   hd, hd, 0);
    // sdpa
    sdpa(ao, q, k, v, seq, NHEAD, nkv, hd, is_full(L)?0:SWIN, 1.0f, 0);
    CU(cudaDeviceSynchronize());
    // o_proj -> tmp, post_attention_layernorm, residual add
    float* op; CU(cudaMalloc(&op,(size_t)seq*H*4));
    linear(m, op, ao, P+"self_attn.o_proj", seq, H, qd);
    rmsnorm(op, op, m.dptr<const uint16_t*>(P+"post_attention_layernorm.weight"), seq, H, EPS, 0);
    add_inplace(h, op, seq*H, 0);     // h = residual + post_attn(attn)
    CU(cudaDeviceSynchronize());
    cudaFree(hn);cudaFree(q);cudaFree(k);cudaFree(v);cudaFree(ao);cudaFree(op);cudaFree(dc);cudaFree(ds);
}

// host fp32 copy of a bf16 tensor
static std::vector<float> bf16_host(Model& m,const std::string& n){
    const auto& t=m.T(n); int d=t.numel(); std::vector<float> o(d); const uint16_t* b=(const uint16_t*)t.data;
    for(int i=0;i<d;++i){ unsigned u=(unsigned)b[i]<<16; memcpy(&o[i],&u,4);} return o; }

// expert FFN for a gathered set of |T| tokens at expert e: out += down(gelu(gate(Xe))*up(Xe)) * w
static void expert_ffn(Model& m,const std::string& EP,float* Xe,int nt,float* moe_out,const int* didx,const float* dw){
    float *g,*u,*dn; CU(cudaMalloc(&g,(size_t)nt*MOE_INT*4)); CU(cudaMalloc(&u,(size_t)nt*MOE_INT*4)); CU(cudaMalloc(&dn,(size_t)nt*H*4));
    linear(m, g, Xe, EP+"gate_proj", nt, MOE_INT, H);
    linear(m, u, Xe, EP+"up_proj",   nt, MOE_INT, H);
    gelu_tanh(g, g, nt*MOE_INT, 0); k_mul<<<(nt*MOE_INT+255)/256,256>>>(g,u,nt*MOE_INT);
    linear(m, dn, g, EP+"down_proj", nt, H, MOE_INT);
    k_scatter_add<<<(nt*H+255)/256,256>>>(moe_out, dn, didx, dw, nt, H);
    CU(cudaDeviceSynchronize()); cudaFree(g);cudaFree(u);cudaFree(dn);
}

static void moe(Model& m, float* h, int seq, int L){
    std::string P="model.language_model.layers."+std::to_string(L)+".";
    float* resid; CU(cudaMalloc(&resid,(size_t)seq*H*4)); CU(cudaMemcpy(resid,h,(size_t)seq*H*4,cudaMemcpyDeviceToDevice));
    // dense MLP: hs1 = post_ff_norm_1( mlp(pre_ff_norm(resid)) )
    float *mi,*g,*u,*hs1; CU(cudaMalloc(&mi,(size_t)seq*H*4));CU(cudaMalloc(&g,(size_t)seq*MLP_INT*4));
    CU(cudaMalloc(&u,(size_t)seq*MLP_INT*4));CU(cudaMalloc(&hs1,(size_t)seq*H*4));
    rmsnorm(mi, resid, m.dptr<const uint16_t*>(P+"pre_feedforward_layernorm.weight"), seq, H, EPS, 0);
    linear(m,g,mi,P+"mlp.gate_proj",seq,MLP_INT,H); linear(m,u,mi,P+"mlp.up_proj",seq,MLP_INT,H);
    gelu_tanh(g,g,seq*MLP_INT,0); k_mul<<<(seq*MLP_INT+255)/256,256>>>(g,u,seq*MLP_INT);
    linear(m,hs1,g,P+"mlp.down_proj",seq,H,MLP_INT);
    rmsnorm(hs1,hs1,m.dptr<const uint16_t*>(P+"post_feedforward_layernorm_1.weight"),seq,H,EPS,0);
    // router on resid (host)
    std::vector<float> rh(seq*H); CU(cudaMemcpy(rh.data(),resid,(size_t)seq*H*4,cudaMemcpyDeviceToHost));
    auto rscale=bf16_host(m,P+"router.scale"); auto rproj=bf16_host(m,P+"router.proj.weight"); auto pes=bf16_host(m,P+"router.per_expert_scale");
    double root=1.0/sqrt((double)H);
    std::vector<std::vector<std::pair<int,float>>> route(seq);
    for(int t=0;t<seq;++t){
        const float* x=&rh[(size_t)t*H]; double ms=0; for(int i=0;i<H;++i)ms+=(double)x[i]*x[i]; double inv=1.0/sqrt(ms/H+EPS);
        std::vector<float> hn(H); for(int i=0;i<H;++i)hn[i]=(float)(x[i]*inv*rscale[i]*root);
        std::vector<float> sc(NEXP); for(int e=0;e<NEXP;++e){ double a=0; const float* w=&rproj[(size_t)e*H]; for(int i=0;i<H;++i)a+=hn[i]*w[i]; sc[e]=(float)a; }
        double mx=-1e30; for(float v:sc)mx=std::max(mx,(double)v); double Z=0; for(int e=0;e<NEXP;++e){sc[e]=expf(sc[e]-mx);Z+=sc[e];} for(int e=0;e<NEXP;++e)sc[e]/=Z;
        std::vector<int> idx(NEXP); for(int e=0;e<NEXP;++e)idx[e]=e;
        std::partial_sort(idx.begin(),idx.begin()+TOPK,idx.end(),[&](int a,int b){return sc[a]>sc[b];});
        double s=0; for(int j=0;j<TOPK;++j)s+=sc[idx[j]];
        for(int j=0;j<TOPK;++j){ float w=(float)(sc[idx[j]]/s)*pes[idx[j]]; route[t].push_back({idx[j],w}); }
    }
    // x2 = pre_ff_norm_2(resid); experts
    float* x2; CU(cudaMalloc(&x2,(size_t)seq*H*4));
    rmsnorm(x2,resid,m.dptr<const uint16_t*>(P+"pre_feedforward_layernorm_2.weight"),seq,H,EPS,0);
    float* moe_out; CU(cudaMalloc(&moe_out,(size_t)seq*H*4)); CU(cudaMemset(moe_out,0,(size_t)seq*H*4));
    // group tokens by expert
    std::vector<std::vector<std::pair<int,float>>> byexp(NEXP); // (token, weight)
    for(int t=0;t<seq;++t) for(auto&pr:route[t]) byexp[pr.first].push_back({t,pr.second});
    float* Xe; CU(cudaMalloc(&Xe,(size_t)seq*H*4)); int *didx; float* dw; CU(cudaMalloc(&didx,seq*4)); CU(cudaMalloc(&dw,seq*4));
    for(int e=0;e<NEXP;++e){ int nt=byexp[e].size(); if(!nt)continue;
        std::vector<int> ti(nt); std::vector<float> tw(nt); for(int j=0;j<nt;++j){ti[j]=byexp[e][j].first;tw[j]=byexp[e][j].second;}
        CU(cudaMemcpy(didx,ti.data(),nt*4,cudaMemcpyHostToDevice)); CU(cudaMemcpy(dw,tw.data(),nt*4,cudaMemcpyHostToDevice));
        k_gather<<<(nt*H+255)/256,256>>>(Xe,x2,didx,nt,H); CU(cudaDeviceSynchronize());
        expert_ffn(m, P+"experts."+std::to_string(e)+".", Xe, nt, moe_out, didx, dw);
    }
    // hs2 = post_ff_norm_2(moe_out); combine; post_ff_norm; residual; layer_scalar
    rmsnorm(moe_out,moe_out,m.dptr<const uint16_t*>(P+"post_feedforward_layernorm_2.weight"),seq,H,EPS,0);
    add_inplace(hs1, moe_out, seq*H, 0);  // hs1 = hs1 + hs2
    rmsnorm(hs1, hs1, m.dptr<const uint16_t*>(P+"post_feedforward_layernorm.weight"), seq, H, EPS, 0);
    add_inplace(resid, hs1, seq*H, 0);    // resid = resid + post_ff(combined)
    // h = resid * layer_scalar  (per-layer learned scalar)
    float lscalar = m.scalarBf16(P+"layer_scalar");
    float* sc1; CU(cudaMalloc(&sc1,seq*4)); std::vector<float> sv(seq,lscalar);
    CU(cudaMemcpy(sc1,sv.data(),seq*4,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(h, resid, (size_t)seq*H*4, cudaMemcpyDeviceToDevice));
    k_scale_rows<<<(seq*H+255)/256,256>>>(h, sc1, seq, H); CU(cudaDeviceSynchronize()); cudaFree(sc1);
    cudaFree(resid);cudaFree(mi);cudaFree(g);cudaFree(u);cudaFree(hs1);cudaFree(x2);cudaFree(moe_out);cudaFree(Xe);cudaFree(didx);cudaFree(dw);
}

static bool DUMP=false;
static void dump_h(float* h,int seq,int idx){
    if(!DUMP)return; std::vector<float> hh((size_t)seq*H); CU(cudaMemcpy(hh.data(),h,(size_t)seq*H*4,cudaMemcpyDeviceToHost));
    char p[128]; snprintf(p,128,"/tmp/cudahs/hs_%d.bin",idx); FILE* f=fopen(p,"wb"); fwrite(hh.data(),4,hh.size(),f); fclose(f);
}
int main(int argc,char**argv){
    if(getenv("DUMP")) DUMP=true; system("mkdir -p /tmp/cudahs");
    std::string ckpt = argc>1?argv[1]:std::string(getenv("HOME"))+"/models/gemma-4-26B-A4B-it-NVFP4/model.safetensors";
    std::string tokfile = argc>2?argv[2]:"/tmp/tokens.txt";
    Model m(ckpt); SC=new Scratch();
    // read token ids
    std::vector<int> ids; { FILE* f=fopen(tokfile.c_str(),"r"); int t; while(fscanf(f,"%d",&t)==1)ids.push_back(t); fclose(f);}
    int seq=ids.size(); printf("seq=%d tokens\n",seq);
    int* dids; CU(cudaMalloc(&dids,seq*4)); CU(cudaMemcpy(dids,ids.data(),seq*4,cudaMemcpyHostToDevice));
    float* h; CU(cudaMalloc(&h,(size_t)seq*H*4));
    k_embed<<<seq,256>>>(h, m.dptr<const uint16_t*>("model.language_model.embed_tokens.weight"), dids, seq, H, EMB_SCALE);
    CU(cudaDeviceSynchronize());
    dump_h(h, seq, 0);
    for(int L=0; L<NLAYER; ++L){ attention(m, h, seq, L); moe(m, h, seq, L); dump_h(h, seq, L+1); }
    // final norm on last token + lm_head (tied embeddings) + softcap
    float* hl; CU(cudaMalloc(&hl,H*4));
    rmsnorm(hl, h+(size_t)(seq-1)*H, m.dptr<const uint16_t*>("model.language_model.norm.weight"), 1, H, EPS, 0);
    float* dlog; CU(cudaMalloc(&dlog,(size_t)VOCAB*4));
    k_lmhead<<<VOCAB,256>>>(dlog, hl, m.dptr<const uint16_t*>("model.language_model.embed_tokens.weight"), H, VOCAB, SOFTCAP);
    CU(cudaDeviceSynchronize());
    std::vector<float> logits(VOCAB); CU(cudaMemcpy(logits.data(),dlog,(size_t)VOCAB*4,cudaMemcpyDeviceToHost));
    double mx=-1e30; for(float v:logits)mx=std::max(mx,(double)v); double Z=0; for(float v:logits)Z+=exp((double)v-mx);
    std::vector<int> ord(VOCAB); for(int i=0;i<VOCAB;++i)ord[i]=i;
    std::partial_sort(ord.begin(),ord.begin()+5,ord.end(),[&](int a,int b){return logits[a]>logits[b];});
    printf("\ntop-5 next-token  (id : logit : logprob):\n");
    for(int j=0;j<5;++j){ int id=ord[j]; double lp=(double)logits[id]-mx-log(Z); printf("  %6d : %8.4f : %8.4f\n",id,logits[id],lp); }
    return 0;
}
