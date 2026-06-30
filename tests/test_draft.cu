// test_draft.cu — structural test for DFlash draft_propose.
// Loads the draft model, builds a dummy taps buffer for C=8 context positions,
// loads the shared target embed_tokens table (bf16) onto device, calls
// draft_propose for k=15, prints the 15 draft ids and validates their range.
#include "draft.h"
#include "safetensors.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

int main(int argc,char**argv){
    std::string home = getenv("HOME");
    std::string draft_path = argc>1?argv[1]:home+"/models/gemma-4-26B-A4B-DFlash/model.safetensors";
    std::string target_path= argc>2?argv[2]:home+"/models/gemma-4-26B-A4B-it-NVFP4/model.safetensors";
    const int H=2816, VOCAB=262144, C=8, K=15;

    // load draft model
    DraftModel* d = draft_load(draft_path.c_str(), /*cap=*/1024);

    // load shared target embed table (== lm_head, tied) -> device bf16 [VOCAB,H]
    st::SafeTensors tgt(target_path);
    const st::Tensor& emb = tgt.get("model.language_model.embed_tokens.weight");
    printf("embed: dtype=%s shape=[%lld,%lld] %.2f GB\n", emb.dtype.c_str(),
           (long long)emb.shape[0], (long long)emb.shape[1], emb.nbytes/1e9);
    if(emb.dtype!="BF16" || emb.shape[0]!=VOCAB || emb.shape[1]!=H){ fprintf(stderr,"unexpected embed\n"); return 1; }
    uint16_t* embed_dev; CU(cudaMalloc(&embed_dev, emb.nbytes));
    CU(cudaMemcpy(embed_dev, emb.data, emb.nbytes, cudaMemcpyHostToDevice));

    // dummy taps [C,6,H] (structural): small deterministic values
    std::vector<float> taps((size_t)C*6*H);
    for(size_t i=0;i<taps.size();++i) taps[i] = 0.02f*(((i*1103515245u+12345u)>>16)&0x7fff)/32768.f - 0.01f;
    float* taps_dev; CU(cudaMalloc(&taps_dev,taps.size()*4));
    CU(cudaMemcpy(taps_dev,taps.data(),taps.size()*4,cudaMemcpyHostToDevice));

    std::vector<int> ctx_pos(C); for(int i=0;i<C;++i) ctx_pos[i]=i;   // positions 0..7
    int next_token = 1000;   // the bonus token at P0 = 8

    std::vector<int> out_ids(K,-1);
    draft_propose(d, taps_dev, ctx_pos.data(), C, next_token, embed_dev, embed_dev, out_ids.data(), K);

    printf("draft ids (k=%d): ", K);
    bool ok=true;
    for(int i=0;i<K;++i){ printf("%d ", out_ids[i]); if(out_ids[i]<0||out_ids[i]>=VOCAB) ok=false; }
    printf("\n%s: %d ids, all in [0,%d)\n", ok?"PASS":"FAIL", K, VOCAB);

    cudaFree(embed_dev); cudaFree(taps_dev); draft_free(d);
    return ok?0:1;
}
