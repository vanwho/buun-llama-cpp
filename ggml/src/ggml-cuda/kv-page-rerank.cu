#include "kv-page-select.cuh"
#include "fattn-common.cuh"
#include "ggml-impl.h"

#include <cmath>

namespace {

// Eight warps own independent rows. Each warp keeps an online LSE, so the row
// loop has no block-wide barriers; the block combines eight partial states once.
__global__ void kv_page_rerank_kernel(
        const float * probes,size_t q0,size_t q1,size_t q2,const char * resident,size_t rk1,size_t rk2,int rr,int rs,
        const char * staged,size_t sk1,size_t sk2,int sr,int ss,const int64_t * descriptors,size_t dn1,
        const int64_t * identity,const int64_t * validity,float * state,size_t st0,size_t st1,size_t st2,size_t st3,
        float * output,size_t out0,size_t out1,size_t out2,int pages,int heads,int nq,int dim,int kvheads,float scale,float cap){
    const int page=blockIdx.x,kh=blockIdx.y,group=heads/kvheads;
    const int64_t * d=(const int64_t *)((const char *)descriptors+page*dn1),*expected=identity+2*(page+1);
    const int64_t slot=d[1],rows=d[2],stream=d[3],psz=d[4],first=d[8],set=d[9];
    const char * keys=set==0?resident:staged;const size_t k1=set==0?rk1:sk1,k2=set==0?rk2:sk2;
    const int nrows=set==0?rr:sr,nstreams=set==0?rs:ss;
    const bool eligible=d[7]&&d[0]>=0&&slot>=0&&rows>0&&(set==0||set==1)&&psz>0&&rows<=psz&&stream>=0&&stream<nstreams&&slot<=(nrows-rows)/psz&&
        d[5]==expected[0]&&d[6]==expected[1]&&validity[0]==identity[0];
    constexpr int MAX_OUT=64, MAX_D=256;const int nout=group*nq;
    if(threadIdx.x==0)for(int o=0;o<nout;++o){int h=kh*group+o/nq,q=o%nq;*(float *)((char *)output+page*out0+h*out1+q*out2)=-INFINITY;}
    if(!eligible)return;
    __shared__ float part_m[MAX_OUT*8],part_s[MAX_OUT*8];
    const int warp=threadIdx.x>>5,lane=threadIdx.x&31;float lm[MAX_OUT],ls[MAX_OUT];
    if(lane==0)for(int o=0;o<nout;++o){lm[o]=-INFINITY;ls[o]=0;}
    for(int64_t row=warp;row<rows;row+=8){
        const char * rd=keys+stream*k2+(slot*psz+row)*k1;float cached[MAX_D/32];
        for(int j=0;j<dim/32;++j){int x=lane+j*32,el=kh*dim+x;const block_turbo4_0 * block=(const block_turbo4_0 *)rd+el/QK_TURBO4;int within=el%QK_TURBO4;uint8_t ix=(within&1)?(block->qs[within/2]>>4):(block->qs[within/2]&15);cached[j]=d_turbo_centroids_4bit_fattn[ix]*__half2float(block->norm);}
        for(int o=0;o<nout;++o){int h=kh*group+o/nq,q=o%nq;if(first+row>validity[1+q]||!validity[5+q])continue;float dot=0;
            for(int j=0;j<dim/32;++j){int x=lane+j*32;float qv=*(const float *)((const char *)probes+x*q0+h*q1+q*q2);dot+=qv*cached[j];}
            for(int delta=16;delta>0;delta>>=1)dot+=__shfl_down_sync(0xffffffff,dot,delta);
            if(lane==0){float z=dot*scale;if(cap>0)z=cap*tanhf(z/cap);if(isfinite(z)){if(z>lm[o]){ls[o]=ls[o]*expf(lm[o]-z)+1;lm[o]=z;}else ls[o]+=expf(z-lm[o]);}}
        }
    }
    if(lane==0)for(int o=0;o<nout;++o){part_m[o*8+warp]=lm[o];part_s[o*8+warp]=ls[o];}
    __syncthreads();
    if(threadIdx.x==0)for(int o=0;o<nout;++o){int h=kh*group+o/nq,q=o%nq;float cm=-INFINITY,cs=0;
        for(int w=0;w<8;++w)if(part_s[o*8+w]>0){if(part_m[o*8+w]>cm){cs=cs*expf(cm-part_m[o*8+w])+part_s[o*8+w];cm=part_m[o*8+w];}else cs+=part_s[o*8+w]*expf(part_m[o*8+w]-cm);}
        float * sm=(float *)((char *)state+page*st1+h*st2+q*st3),*sv=(float *)((char *)state+st0+page*st1+h*st2+q*st3);float om=*sm,ov=*sv;
        if(cs>0||ov>0){float m=fmaxf(om,cm),sum=(ov>0?ov*expf(om-m):0)+(cs>0?cs*expf(cm-m):0);if(sum>0&&isfinite(m)&&isfinite(sum)){*sm=m;*sv=sum;*(float *)((char *)output+page*out0+h*out1+q*out2)=m+logf(sum);}}
    }
}

static __device__ bool mass_eligible(int page,const char * descriptors,size_t desc_nb1,const int64_t * ids,const int64_t * valid){
    const int64_t * d=(const int64_t *)(descriptors+page*desc_nb1),* e=ids+2*(page+1);
    return d[7]&&d[0]>=0&&d[5]==e[0]&&d[6]==e[1]&&valid[0]==ids[0];
}

__global__ void kv_page_mass_normalize(const float * state,size_t nb0,size_t nb1,size_t nb2,size_t nb3,
        const char * desc,size_t desc_nb1,const int64_t * ids,const int64_t * valid,float * probs,
        int pages,int heads,int probes){
    const int channel=blockIdx.x, h=channel%heads, q=channel/heads;
    __shared__ float scratch[256];
    float local_max=-INFINITY;
    for(int p=threadIdx.x;p<pages;p+=blockDim.x){
        if(!valid[5+q]||!mass_eligible(p,desc,desc_nb1,ids,valid))continue;
        const float m=*(const float *)((const char *)state+p*nb1+h*nb2+q*nb3);
        const float s=*(const float *)((const char *)state+nb0+p*nb1+h*nb2+q*nb3);
        if(s>0&&isfinite(s)&&isfinite(m))local_max=fmaxf(local_max,m+logf(s));
    }
    scratch[threadIdx.x]=local_max;__syncthreads();
    for(int step=128;step>0;step>>=1){if(threadIdx.x<step)scratch[threadIdx.x]=fmaxf(scratch[threadIdx.x],scratch[threadIdx.x+step]);__syncthreads();}
    const float maximum=scratch[0];
    float local_sum=0.0f;
    if(isfinite(maximum))for(int p=threadIdx.x;p<pages;p+=blockDim.x){
        if(!valid[5+q]||!mass_eligible(p,desc,desc_nb1,ids,valid))continue;
        const float m=*(const float *)((const char *)state+p*nb1+h*nb2+q*nb3);
        const float s=*(const float *)((const char *)state+nb0+p*nb1+h*nb2+q*nb3);
        if(s>0&&isfinite(s)&&isfinite(m))local_sum+=expf(m+logf(s)-maximum);
    }
    scratch[threadIdx.x]=local_sum;__syncthreads();
    for(int step=128;step>0;step>>=1){if(threadIdx.x<step)scratch[threadIdx.x]+=scratch[threadIdx.x+step];__syncthreads();}
    const float denom=scratch[0];
    for(int p=threadIdx.x;p<pages;p+=blockDim.x){
        float probability=0.0f;
        if(isfinite(maximum)&&denom>0&&valid[5+q]&&mass_eligible(p,desc,desc_nb1,ids,valid)){
            const float m=*(const float *)((const char *)state+p*nb1+h*nb2+q*nb3);
            const float s=*(const float *)((const char *)state+nb0+p*nb1+h*nb2+q*nb3);
            if(s>0&&isfinite(s)&&isfinite(m))probability=expf(m+logf(s)-maximum)/denom;
        }
        probs[p+pages*channel]=probability;
    }
}

__global__ void kv_page_mass_emit(const float * state,size_t nb0,size_t nb1,size_t nb2,size_t nb3,
        const char * desc,size_t desc_nb1,const int64_t * ids,const int64_t * valid,const float * probs,
        ggml_kv_page_rank_record * output,int pages,int heads,int probes){
    const int page=blockIdx.x, tid=threadIdx.x;
    __shared__ float peak[256],sum[256];__shared__ int count[256];
    float pmax=0,psum=0;int n=0;
    for(int channel=tid;channel<heads*probes;channel+=blockDim.x){
        const int q=channel/heads,h=channel%heads; if(!valid[5+q])continue;
        const float m=*(const float *)((const char *)state+page*nb1+h*nb2+q*nb3);
        const float z=*(const float *)((const char *)state+nb0+page*nb1+h*nb2+q*nb3);
        const float x=probs[page+pages*channel];
        // The mean denominator is the set of globally valid channels, not only
        // channels for which this page had a causal row. A missing contribution
        // is zero but still participates in the mean. If no channel has a
        // finite denominator anywhere, all means stay zero and owner validation
        // fails the layer normalization check.
        if(valid[5+q]){
            ++n;
            if(z>0&&isfinite(z)&&isfinite(m)){
                const float bounded=fminf(1.0f,fmaxf(0.0f,x));
                pmax=fmaxf(pmax,bounded);
                psum+=bounded;
            }
        }
    }
    peak[tid]=pmax;sum[tid]=psum;count[tid]=n;__syncthreads();
    for(int step=128;step>0;step>>=1){if(tid<step){peak[tid]=fmaxf(peak[tid],peak[tid+step]);sum[tid]+=sum[tid+step];count[tid]+=count[tid+step];}__syncthreads();}
    if(tid==0){
        const int64_t * d=(const int64_t *)(desc+page*desc_nb1);
        if(mass_eligible(page,desc,desc_nb1,ids,valid)&&count[0]>0)
            output[page]={int32_t(d[0]),1u,peak[0],sum[0]/count[0]};
        else output[page]={-1,0u,0.0f,0.0f};
    }
}

} // namespace

void ggml_cuda_op_kv_page_rerank(ggml_backend_cuda_context & ctx,ggml_tensor * dst){
    const ggml_tensor * q=dst->src[0],*rk=dst->src[1],*sk=dst->src[2],*desc=dst->src[3],*ids=dst->src[4],*valid=dst->src[5],*state=dst->src[6];
    const int pages=int(dst->ne[0]),heads=int(dst->ne[1]),probes=int(dst->ne[2]),dim=int(q->ne[0]),kvheads=int(rk->ne[0]/dim);
    kv_page_rerank_kernel<<<dim3(pages,kvheads,1),256,0,ctx.stream()>>>((const float *)q->data,q->nb[0],q->nb[1],q->nb[2],
        (const char *)rk->data,rk->nb[1],rk->nb[2],int(rk->ne[1]),int(rk->ne[2]),(const char *)sk->data,sk->nb[1],sk->nb[2],int(sk->ne[1]),int(sk->ne[2]),
        (const int64_t *)desc->data,desc->nb[1],(const int64_t *)ids->data,(const int64_t *)valid->data,(float *)state->data,state->nb[0],state->nb[1],state->nb[2],state->nb[3],
        (float *)dst->data,dst->nb[0],dst->nb[1],dst->nb[2],pages,heads,probes,dim,kvheads,ggml_get_op_params_f32(dst,0),ggml_get_op_params_f32(dst,1));
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_kv_page_mass(ggml_backend_cuda_context & ctx,ggml_tensor * dst){
    const ggml_tensor * state=dst->src[0],*desc=dst->src[1],*ids=dst->src[2],*valid=dst->src[3];
    const int pages=int(state->ne[1]),heads=int(state->ne[2]),probes=int(state->ne[3]);
    ggml_cuda_pool_alloc<float> probs(ctx.pool(),size_t(pages)*heads*probes);
    kv_page_mass_normalize<<<heads*probes,256,0,ctx.stream()>>>((const float *)state->data,state->nb[0],state->nb[1],state->nb[2],state->nb[3],
        (const char *)desc->data,desc->nb[1],(const int64_t *)ids->data,(const int64_t *)valid->data,probs.get(),pages,heads,probes);
    CUDA_CHECK(cudaGetLastError());
    kv_page_mass_emit<<<pages,256,0,ctx.stream()>>>((const float *)state->data,state->nb[0],state->nb[1],state->nb[2],state->nb[3],
        (const char *)desc->data,desc->nb[1],(const int64_t *)ids->data,(const int64_t *)valid->data,probs.get(),
        (ggml_kv_page_rank_record *)dst->data,pages,heads,probes);
    CUDA_CHECK(cudaGetLastError());
}
