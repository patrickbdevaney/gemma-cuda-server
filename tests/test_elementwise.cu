// test_elementwise.cu — micro-gate for rmsnorm + rotate-half RoPE vs numpy.
#include "elementwise.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
static std::vector<uint8_t> rd(const std::string& p){FILE*f=fopen(p.c_str(),"rb");if(!f){printf("open %s\n",p.c_str());exit(1);}fseek(f,0,2);long n=ftell(f);fseek(f,0,0);std::vector<uint8_t>v(n);if(fread(v.data(),1,n,f)!=(size_t)n)exit(1);fclose(f);return v;}
#define CU(x) do{cudaError_t e=(x);if(e){printf("cuda %d %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static double cmp(const float*a,const float*b,int n){double m=0;for(int i=0;i<n;++i)m=fmax(m,fabs((double)a[i]-b[i]));return m;}

int main(int argc,char**argv){
    std::string d=argc>1?argv[1]:"/tmp/ewcase";
    int ROWS,DIM,HEADS,HD,ROT;
    {FILE*f=fopen((d+"/dims.txt").c_str(),"r");fscanf(f,"%d %d %d %d %d",&ROWS,&DIM,&HEADS,&HD,&ROT);fclose(f);}

    // rmsnorm
    auto x=rd(d+"/rn_x.bin"); auto w=rd(d+"/rn_w.bin"); auto rf=rd(d+"/rn_ref.bin");
    float*dx,*dout; uint16_t*dw;
    CU(cudaMalloc(&dx,x.size()));CU(cudaMalloc(&dout,x.size()));CU(cudaMalloc(&dw,w.size()));
    CU(cudaMemcpy(dx,x.data(),x.size(),cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));
    rmsnorm(dout,dx,dw,ROWS,DIM,1e-6f,0); CU(cudaDeviceSynchronize());
    std::vector<float> o(ROWS*DIM); CU(cudaMemcpy(o.data(),dout,o.size()*4,cudaMemcpyDeviceToHost));
    double rn_err=cmp(o.data(),(const float*)rf.data(),ROWS*DIM);
    printf("rmsnorm max_abs=%.3e\n",rn_err);

    // rope
    auto rx=rd(d+"/rope_x.bin");auto rc=rd(d+"/rope_cos.bin");auto rs=rd(d+"/rope_sin.bin");auto rr=rd(d+"/rope_ref.bin");
    float*drx,*drc,*drs; CU(cudaMalloc(&drx,rx.size()));CU(cudaMalloc(&drc,rc.size()));CU(cudaMalloc(&drs,rs.size()));
    CU(cudaMemcpy(drx,rx.data(),rx.size(),cudaMemcpyHostToDevice));
    CU(cudaMemcpy(drc,rc.data(),rc.size(),cudaMemcpyHostToDevice));
    CU(cudaMemcpy(drs,rs.data(),rs.size(),cudaMemcpyHostToDevice));
    rope_rotate_half(drx,drc,drs,ROWS,HEADS,HD,ROT,0); CU(cudaDeviceSynchronize());
    std::vector<float> ro(ROWS*HEADS*HD); CU(cudaMemcpy(ro.data(),drx,ro.size()*4,cudaMemcpyDeviceToHost));
    double rope_err=cmp(ro.data(),(const float*)rr.data(),ROWS*HEADS*HD);
    printf("rope    max_abs=%.3e\n",rope_err);

    bool pass = rn_err<2e-3 && rope_err<2e-5;
    printf("%s\n", pass?"GATE PASS ✅":"GATE FAIL ❌");
    return pass?0:2;
}
