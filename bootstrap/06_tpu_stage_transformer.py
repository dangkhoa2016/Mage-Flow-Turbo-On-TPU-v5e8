#!/usr/bin/env python3
import argparse, hashlib, json, resource, sys, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

def parse_topology(value):
    r,m=(int(x) for x in value.lower().split("x",1))
    if r*m != 8 or (r,m) not in {(1,8),(2,4),(4,2)}:
        raise ValueError("supported topologies: 1x8, 2x4, 4x2")
    return r,m

def main():
    ap=argparse.ArgumentParser()
    for k in ("runtime-root","source-root","checkpoint","basis-dim16","basis-dim56","conditioning","output"):
        ap.add_argument("--"+k,required=True)
    ap.add_argument("--topology",default="4x2")
    ap.add_argument("--resolution",type=int,choices=(512,768,1024),default=512)
    ap.add_argument("--steps",type=int,default=4)
    ap.add_argument("--seeds",required=True)
    ap.add_argument("--attention-mode",choices=("masked-global","segmented","segmented-query-chunk"),required=True)
    ap.add_argument("--query-chunk",type=int,default=256)
    args=ap.parse_args()
    replicas,model_axis=parse_topology(args.topology)
    seeds=[int(x) for x in args.seeds.split(",") if x.strip()]
    if len(seeds)!=replicas: raise SystemExit("seed count must equal replica count")
    if args.steps!=4: raise SystemExit("only qualified 4-step schedule is supported")
    sys.path.insert(0,args.runtime_root)
    from experiment import rope_provider, transformer_driver
    from experiment.runtime_contract import SIGMA_SCHEDULE
    if jax.default_backend()!="tpu" or len(jax.devices())!=8: raise SystemExit("TPU/8 required")
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    C=128; H=W=args.resolution//16; n=H*W
    mesh=Mesh(np.asarray(jax.devices()).reshape(replicas,model_axis),("replica","model"))
    cpu=jax.devices("cpu")[0]
    cond=np.load(args.conditioning).astype(np.float32)
    if cond.shape!=(1,19,2560): raise SystemExit(f"bad conditioning shape {cond.shape}")
    txt=jax.device_put(np.concatenate([cond]*replicas,axis=1),NamedSharding(mesh,P(None,"replica",None)))
    with jax.default_device(cpu):
        source=transformer_driver.load_transformer_source(args.source_root)
        model=transformer_driver.construct_transformer_model(source)
        restored=transformer_driver.restore_transformer_checkpoint(args.checkpoint)
    paths=model.parameter_paths()
    if len(paths)!=397 or set(restored)!=set(paths): raise SystemExit("transformer key contract failed")
    plan=transformer_driver.static_sharding_plan(model,mesh_size=model_axis,expect_contract=True)
    for key,var in paths.items():
        spec=plan["plan"][key]
        host=np.asarray(jax.device_get(restored[key])).astype(np.float32,copy=False)
        var.assign(jax.device_put(host,NamedSharding(mesh,P(*spec))))
    del restored
    img_cu=jax.device_put(np.array([i*n for i in range(replicas+1)],np.int32),NamedSharding(mesh,P()))
    txt_cu=jax.device_put(np.array([i*19 for i in range(replicas+1)],np.int32),NamedSharding(mesh,P()))
    rope_one=rope_provider.build_image_rope("control",H,W,args.basis_dim16,args.basis_dim56)
    if rope_one.shape!=(n,64,2): raise SystemExit(f"bad rope shape {rope_one.shape}")
    rope=jax.device_put(np.concatenate([rope_one]*replicas,axis=0),NamedSharding(mesh,P("replica",None,None)))
    latent_sh=NamedSharding(mesh,P("replica",None,None,None))
    token_sh=NamedSharding(mesh,P(None,"replica",None))
    timestep_sh=NamedSharding(mesh,P("replica"))
    attn_kwargs={"attention_mode":args.attention_mode,"query_chunk":args.query_chunk}

    def make_x():
        xs=[jax.random.normal(jax.random.PRNGKey(seed),(1,C,H,W),dtype=jnp.float32) for seed in seeds]
        return jax.device_put(jnp.concatenate(xs,axis=0),latent_sh)

    def run_once():
        x=make_x(); step_s=[]; wall=time.time()
        for i in range(args.steps):
            sigma=float(SIGMA_SCHEDULE[i]); sigma_next=float(SIGMA_SCHEDULE[i+1])
            tokens=jnp.transpose(x,(0,2,3,1)).reshape(1,replicas*n,C)
            tokens=jax.device_put(tokens,token_sh)
            ts=jax.device_put(np.full((replicas,),sigma,np.float32),timestep_sh)
            t=time.time()
            velocity_packed=model.forward(tokens,txt,ts,rope,img_cu,txt_cu,joint_attention_kwargs=attn_kwargs)
            velocity=jnp.transpose(velocity_packed.reshape(replicas,H,W,C),(0,3,1,2))
            x=jax.device_put(x+np.float32(sigma_next-sigma)*velocity,latent_sh)
            jax.block_until_ready(x); step_s.append(time.time()-t)
        return np.asarray(jax.device_get(x)),step_s,time.time()-wall
    cold,cold_steps,cold_wall=run_once()
    if cold.shape!=(replicas,C,H,W) or not np.isfinite(cold).all(): raise SystemExit("cold output gate failed")
    warm,warm_steps,warm_wall=run_once()
    if not np.array_equal(cold,warm): raise SystemExit("warm rerun is not bit-exact")
    hashes=[]
    for i,seed in enumerate(seeds):
        arr=cold[i:i+1]
        np.save(out/f"C5_seed_{seed}.npy",arr)
        hashes.append(hashlib.sha256(arr.tobytes()).hexdigest())
    if len(set(hashes))!=replicas: raise SystemExit("seed outputs are not unique")
    hbm=[]
    for i,d in enumerate(jax.devices()):
        m=d.memory_stats() or {}
        hbm.append({"device":i,"peak_bytes_in_use":m.get("peak_bytes_in_use"),
                    "bytes_in_use":m.get("bytes_in_use"),"bytes_limit":m.get("bytes_limit")})
    summary={"status":"PASS","stage":"transformer","backend":jax.default_backend(),"device_count":8,
      "topology":args.topology,"mesh_shape":[replicas,model_axis],"resolution":[args.resolution,args.resolution],
      "steps":args.steps,"seeds":seeds,"attention_mode":args.attention_mode,
      "query_chunk":args.query_chunk if args.attention_mode=="segmented-query-chunk" else None,
      "parameter_leaves":len(paths),"parameter_sharded":plan["sharded"],"parameter_replicated":plan["replicated"],
      "cold_wall_s":cold_wall,"cold_step_s":cold_steps,"warm_wall_s":warm_wall,"warm_step_s":warm_steps,
      "warm_effective_s_per_image":warm_wall/replicas,"warm_images_per_min":60.0*replicas/warm_wall,
      "deterministic_bit_exact":True,"outputs_unique":True,"C5_hashes":hashes,
      "host_maxrss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"hbm":hbm}
    (out/"transformer_stage_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    main()
