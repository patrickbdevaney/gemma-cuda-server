// test_attention.cu — micro-gate for single-sequence causal SDPA (GQA + sliding window) vs numpy.
// Each argv is a case dir produced by gen_attention_ref.py. Prints GATE PASS only if ALL pass.
#include "attention.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
static std::vector<uint8_t> rd(const std::string& p){FILE*f=fopen(p.c_str(),"rb");if(!f){printf("open %s\n",p.c_str());exit(1);}fseek(f,0,2);long n=ftell(f);fseek(f,0,0);std::vector<uint8_t>v(n);if(fread(v.data(),1,n,f)!=(size_t)n)exit(1);fclose(f);return v;}
#define CU(x) do{cudaError_t e=(x);if(e){printf("cuda %d %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

static bool run_case(const std::string& d){
    int SEQ,HEADS,NKV,HD,WIN;
    {FILE*f=fopen((d+"/dims.txt").c_str(),"r");if(!f){printf("no dims in %s\n",d.c_str());return false;}fscanf(f,"%d %d %d %d %d",&SEQ,&HEADS,&NKV,&HD,&WIN);fclose(f);}
    auto q=rd(d+"/q.bin"); auto k=rd(d+"/k.bin"); auto v=rd(d+"/v.bin"); auto rf=rd(d+"/out_ref.bin");
    size_t qn=(size_t)SEQ*HEADS*HD, kn=(size_t)SEQ*NKV*HD;
    float *dq,*dk,*dv,*dout;
    CU(cudaMalloc(&dq,qn*4));CU(cudaMalloc(&dk,kn*4));CU(cudaMalloc(&dv,kn*4));CU(cudaMalloc(&dout,qn*4));
    CU(cudaMemcpy(dq,q.data(),qn*4,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dk,k.data(),kn*4,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dv,v.data(),kn*4,cudaMemcpyHostToDevice));
    sdpa(dout,dq,dk,dv,SEQ,HEADS,NKV,HD,WIN,1.0f,0);
    CU(cudaGetLastError()); CU(cudaDeviceSynchronize());
    std::vector<float> o(qn); CU(cudaMemcpy(o.data(),dout,qn*4,cudaMemcpyDeviceToHost));
    const float* ref=(const float*)rf.data();
    double m=0, amax=0;
    for(size_t i=0;i<qn;++i){m=fmax(m,fabs((double)o[i]-ref[i]));amax=fmax(amax,fabs((double)ref[i]));}
    double tol=1e-4*(amax+1.0);
    bool pass=m<tol;
    printf("  %-22s seq=%d nh=%d nkv=%d hd=%d win=%-4d  max_abs=%.3e  tol=%.3e  %s\n",
           d.c_str(),SEQ,HEADS,NKV,HD,WIN,m,tol,pass?"ok":"FAIL");
    cudaFree(dq);cudaFree(dk);cudaFree(dv);cudaFree(dout);
    return pass;
}

int main(int argc,char**argv){
    if(argc<2){printf("usage: %s <case dir> [<case dir> ...]\n",argv[0]);return 2;}
    bool all=true;
    for(int a=1;a<argc;++a) all &= run_case(argv[a]);
    printf("%s\n", all?"GATE PASS ✅":"GATE FAIL ❌");
    return all?0:2;
}
