//! General bounded-workspace attention, normalization and spatial convolution.
use super::gpu_support::StorageType;
use crate::codegen::{
    GpuAssign, GpuDialect, GpuFunction, GpuValue,
    components::{input_components, output_components, single_value, value_expr},
    gpu::GpuRenderError,
    render_utils::{invalid_inputs, invalid_outputs},
};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::codegen) enum DiffusionOp {
    Attention(StorageType),
    Norm(StorageType),
    Conv2d,
    Linear,
}
impl DiffusionOp {
    pub(in crate::codegen) fn from_operation(op: &str) -> Option<Self> {
        Some(match op {
            "materializec.attention-bf16" => Self::Attention(StorageType::Bf16),
            "materializec.attention-f32" => Self::Attention(StorageType::F32),
            "materializec.norm-bf16" => Self::Norm(StorageType::Bf16),
            "materializec.norm-f32" => Self::Norm(StorageType::F32),
            "materializec.conv2d-f32" => Self::Conv2d,
            "materializec.linear-f32" => Self::Linear,
            _ => return None,
        })
    }
    fn inputs(self) -> usize {
        match self {
            Self::Attention(_) => 15,
            Self::Norm(_) => 13,
            Self::Conv2d => 13,
            Self::Linear => 10,
        }
    }
    fn parts<'a>(self, a: &'a GpuAssign) -> Result<Vec<&'a GpuValue>, GpuRenderError> {
        let c = input_components(a)?;
        if c.len() != self.inputs() {
            return Err(invalid_inputs(a, self.inputs()));
        }
        c.iter()
            .map(|c| single_value(c).map_err(|_| invalid_inputs(a, self.inputs())))
            .collect()
    }
    fn name(f: &GpuFunction, a: &GpuAssign) -> Result<String, GpuRenderError> {
        Ok(format!(
            "diffusion_{}_{}",
            f.name,
            a.outputs.last().ok_or_else(|| invalid_outputs(a, 1))?.name
        ))
    }
    pub(in crate::codegen) fn kernels(
        self,
        out: &mut String,
        f: &GpuFunction,
        a: &GpuAssign,
    ) -> Result<(), GpuRenderError> {
        self.parts(a)?;
        let name = Self::name(f, a)?;
        match self {
   Self::Attention(s)=>{
    let ty=s.c_type();let q=s.load("q[(row*heads+head)*dim+c]");let k=s.load("k[(pos*kvheads+head/(heads/kvheads))*dim+c]");let v=s.load("v[(pos*kvheads+head/(heads/kvheads))*dim+c]");let store=s.store("acc[c/256]/denom");
    out.push_str(&format!(r#"
__global__ void {name}(const {ty} *q,const {ty} *k,const {ty} *v,{ty} *o,
 uint64_t nq,uint64_t nk,uint64_t heads,uint64_t kvheads,uint64_t dim,float scale,uint64_t causal,uint64_t valid) {{
 uint64_t row=blockIdx.x,head=blockIdx.y;uint32_t tid=threadIdx.x,lane=tid%32,keylane=tid/32;
 __shared__ float query[1024],scores[8],state[3];float acc[4]={{0,0,0,0}};
 for(uint64_t c=tid;c<dim;c+=256) query[c]={q};
 if(tid==0){{state[0]=-INFINITY;state[1]=0;state[2]=0;}}__syncthreads();
 uint64_t count=valid<nk?valid:nk;if(causal && count>row+1)count=row+1;
 for(uint64_t first=0;first<count;first+=8){{
  uint64_t pos=first+keylane;float dot=0;
  if(pos<count)for(uint64_t c=lane;c<dim;c+=32)dot+=query[c]*{k};
  for(int offset=16;offset>0;offset/=2)dot+=__shfl_down(dot,offset,32);
  if(lane==0)scores[keylane]=pos<count?dot*scale:-INFINITY;__syncthreads();
  if(tid==0){{float m=state[0];for(int j=0;j<8;j++)m=fmaxf(m,scores[j]);
   float factor=expf(state[0]-m);float sum=0;for(int j=0;j<8;j++){{scores[j]=expf(scores[j]-m);sum+=scores[j];}}
   state[1]=state[1]*factor+sum;state[0]=m;state[2]=factor;}}__syncthreads();
  for(uint64_t c=tid;c<dim;c+=256){{float sum=acc[c/256]*state[2];
   for(int j=0;j<8 && first+j<count;j++){{pos=first+j;sum+=scores[j]*{v};}}acc[c/256]=sum;}}__syncthreads();
 }}
 float denom=state[1];for(uint64_t c=tid;c<dim;c+=256)o[(row*heads+head)*dim+c]={store};
}}
"#));
   }
   Self::Norm(s)=>{
    let ty=s.c_type();let load=s.load("x[row*cols+j]");let w=s.load("weight[wi]");let b=s.load("bias[wi]");let store=s.store("value");
    out.push_str(&format!(r#"
__global__ void {name}(const {ty}*x,const {ty}*weight,const {ty}*bias,{ty}*o,
 uint64_t rows,uint64_t cols,uint64_t spatial,uint64_t kind,uint64_t affine,float eps){{
 uint64_t row=blockIdx.x;int tid=threadIdx.x;__shared__ float sums[256];__shared__ float mean,inv;
 float sum=0;for(uint64_t j=tid;j<cols;j+=256)sum+={load};sums[tid]=sum;__syncthreads();
 for(int stride=128;stride;stride/=2){{if(tid<stride)sums[tid]+=sums[tid+stride];__syncthreads();}}
 if(tid==0)mean=kind==1?0:sums[0]/cols;__syncthreads();
 sum=0;for(uint64_t j=tid;j<cols;j+=256){{float d={load}-mean;sum+=d*d;}}sums[tid]=sum;__syncthreads();
 for(int stride=128;stride;stride/=2){{if(tid<stride)sums[tid]+=sums[tid+stride];__syncthreads();}}
 if(tid==0)inv=rsqrtf(sums[0]/cols+eps);__syncthreads();
 for(uint64_t j=tid;j<cols;j+=256){{uint64_t wi=kind==2?(row*cols+j)/spatial:j;
  float value=({load}-mean)*inv;if(affine)value*= {w};if(affine==2)value+={b};o[row*cols+j]={store};}}
}}
"#));
   }
   Self::Conv2d=>out.push_str(&format!(r#"
__global__ void {name}(const float*x,const float*w,const float*b,float*o,
 uint64_t ci,uint64_t co,uint64_t h,uint64_t width,uint64_t ks,uint64_t pad){{
 __shared__ float a[16][16],v[16][16];uint64_t oc=blockIdx.y*16+threadIdx.y;
 uint64_t pixel=blockIdx.x*16+threadIdx.x,spatial=h*width;float acc=0;
 uint64_t terms=ci*ks*ks;
 for(uint64_t base=0;base<terms;base+=16){{
  uint64_t ai=base+threadIdx.x;a[threadIdx.y][threadIdx.x]=(oc<co && ai<terms)?w[oc*terms+ai]:0;
  uint64_t vi=base+threadIdx.y;int64_t yy=(int64_t)(pixel/width)+(int64_t)((vi/ks)%ks)-(int64_t)pad;
  int64_t xx=(int64_t)(pixel%width)+(int64_t)(vi%ks)-(int64_t)pad;
  v[threadIdx.y][threadIdx.x]=(pixel<spatial && vi<terms && yy>=0 && yy<(int64_t)h && xx>=0 && xx<(int64_t)width)?x[(vi/(ks*ks))*spatial+yy*width+xx]:0;
  __syncthreads();for(int j=0;j<16;j++)acc+=a[threadIdx.y][j]*v[j][threadIdx.x];__syncthreads();
 }}if(oc<co && pixel<spatial)o[oc*spatial+pixel]=acc+b[oc];
}}
"#)),
   Self::Linear=>out.push_str(&format!(r#"
__global__ void {name}(const float*x,const float*w,const float*b,float*o,uint64_t rows,uint64_t ci,uint64_t co){{
 __shared__ float a[16][16],v[16][16];uint64_t row=blockIdx.x*16+threadIdx.y,col=blockIdx.y*16+threadIdx.x;float acc=0;
 for(uint64_t base=0;base<ci;base+=16){{uint64_t ai=base+threadIdx.x,vi=base+threadIdx.y;
 a[threadIdx.y][threadIdx.x]=(row<rows && ai<ci)?x[row*ci+ai]:0;
 v[threadIdx.y][threadIdx.x]=(col<co && vi<ci)?w[col*ci+vi]:0;
 __syncthreads();for(int j=0;j<16;j++)acc+=a[threadIdx.y][j]*v[j][threadIdx.x];__syncthreads();}}
 if(row<rows && col<co)o[row*co+col]=acc+b[col];
}}
"#)),
  }
        Ok(())
    }
    pub(in crate::codegen) fn call(
        self,
        out: &mut String,
        f: &GpuFunction,
        a: &GpuAssign,
        d: GpuDialect,
    ) -> Result<(), GpuRenderError> {
        let p = self.parts(a)?;
        let e = |i| value_expr(p[i]);
        let outputs = output_components(a)?;
        if outputs.len() != 1 || outputs[0].len() != 1 {
            return Err(invalid_outputs(a, 1));
        }
        let o = &outputs[0][0].name;
        let name = Self::name(f, a)?;
        let ty = match self {
            Self::Attention(s) | Self::Norm(s) => s.c_type(),
            _ => "float",
        };
        out.push_str(&format!("    {ty}* {o}_data=nullptr;\n    catena_assert(({count})>0 && ({count})<=UINT64_MAX/sizeof({ty}));\n    catena_host_gpu_check({alloc}((void**)&{o}_data,({count})*sizeof({ty}),nullptr));\n",count=e(6),alloc=d.device_alloc_async_fn()));
        let (asserts, grid, block, args): (String, String, String, String) = match self {
            Self::Attention(_) => (
                format!(
                    "catena_assert(({nq})>0 && ({nk})>0 && ({heads})>0 && ({kvheads})>0 && ({heads})%({kvheads})==0 && ({dim})>0 && ({dim})<=1024 && ({valid})>0 && ({valid})<=({nk}));\ncatena_assert(({qc})==({nq})*({heads})*({dim}) && ({kc})==({nk})*({kvheads})*({dim}) && ({vc})==({kc}) && ({oc})==({qc}));",
                    nq = e(7),
                    nk = e(8),
                    heads = e(9),
                    kvheads = e(10),
                    dim = e(11),
                    valid = e(14),
                    qc = e(1),
                    kc = e(3),
                    vc = e(5),
                    oc = e(6)
                ),
                format!("dim3({}, {})", e(7), e(9)),
                "dim3(256)".into(),
                format!(
                    "{},{},{},{o}_data,{},{},{},{},{},{},{},{}",
                    e(0),
                    e(2),
                    e(4),
                    e(7),
                    e(8),
                    e(9),
                    e(10),
                    e(11),
                    e(12),
                    e(13),
                    e(14)
                ),
            ),
            Self::Norm(_) => (
                format!(
                    "catena_assert(({rows})>0 && ({cols})>0 && ({sp})>0 && ({kind})<=2 && ({af})<=2 && ({oc})==({xc}) && ({xc})==({rows})*({cols}));\ncatena_assert(({af})==0 || (({wc})>=(({kind})==2?({xc})/({sp}):({cols})) && (({af})!=2 || ({bc})>=({wc}))));",
                    rows = e(7),
                    cols = e(8),
                    sp = e(9),
                    kind = e(10),
                    af = e(11),
                    oc = e(6),
                    xc = e(1),
                    wc = e(3),
                    bc = e(5)
                ),
                format!("dim3({})", e(7)),
                "dim3(256)".into(),
                format!(
                    "{},{},{},{o}_data,{},{},{},{},{},{}",
                    e(0),
                    e(2),
                    e(4),
                    e(7),
                    e(8),
                    e(9),
                    e(10),
                    e(11),
                    e(12)
                ),
            ),
            Self::Conv2d => (
                format!(
                    "catena_assert(({ci})>0 && ({co})>0 && ({h})>0 && ({w})>0 && (({ks})==1 || ({ks})==3) && ({pad})==(({ks})-1)/2);\ncatena_assert(({xc})==({ci})*({h})*({w}) && ({wc})==({co})*({ci})*({ks})*({ks}) && ({bc})==({co}) && ({oc})==({co})*({h})*({w}));",
                    ci = e(7),
                    co = e(8),
                    h = e(9),
                    w = e(10),
                    ks = e(11),
                    pad = e(12),
                    xc = e(1),
                    wc = e(3),
                    bc = e(5),
                    oc = e(6)
                ),
                format!("dim3(({}*{}+15)/16,({}+15)/16)", e(9), e(10), e(8)),
                "dim3(16,16)".into(),
                format!(
                    "{},{},{},{o}_data,{},{},{},{},{},{}",
                    e(0),
                    e(2),
                    e(4),
                    e(7),
                    e(8),
                    e(9),
                    e(10),
                    e(11),
                    e(12)
                ),
            ),
            Self::Linear => (
                format!(
                    "catena_assert(({rows})>0 && ({ci})>0 && ({co})>0 && ({xc})==({rows})*({ci}) && ({wc})==({co})*({ci}) && ({bc})==({co}) && ({oc})==({rows})*({co}));",
                    rows = e(7),
                    ci = e(8),
                    co = e(9),
                    xc = e(1),
                    wc = e(3),
                    bc = e(5),
                    oc = e(6)
                ),
                format!("dim3(({}+15)/16,({}+15)/16)", e(7), e(9)),
                "dim3(16,16)".into(),
                format!(
                    "{},{},{},{o}_data,{},{},{}",
                    e(0),
                    e(2),
                    e(4),
                    e(7),
                    e(8),
                    e(9)
                ),
            ),
        };
        out.push_str(&format!(
            "    {asserts}\n    {name}<<<{grid},{block}>>>({args});\n    {o}={o}_data;\n"
        ));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn primitive_contracts_are_classified_explicitly() {
        for (name, count) in [
            ("materializec.attention-bf16", 15),
            ("materializec.attention-f32", 15),
            ("materializec.norm-bf16", 13),
            ("materializec.norm-f32", 13),
            ("materializec.conv2d-f32", 13),
            ("materializec.linear-f32", 10),
        ] {
            assert_eq!(DiffusionOp::from_operation(name).unwrap().inputs(), count);
        }
        assert!(DiffusionOp::from_operation("materializec.cached-attention").is_none());
    }
}
