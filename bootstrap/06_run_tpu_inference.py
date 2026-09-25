#!/usr/bin/env python3
import argparse, json, os, subprocess, sys, time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TPU_ENV={
    "PJRT_DEVICE":"TPU",
    "TPU_ACCELERATOR_TYPE":"v5litepod-8",
    "TPU_CHIPS_PER_HOST_BOUNDS":"2,4,1",
    "TPU_HOST_BOUNDS":"1,1,1",
    "TPU_PROCESS_ADDRESSES":"local",
    "TPU_SKIP_MDS_QUERY":"1",
    "TPU_WORKER_HOSTNAMES":"localhost",
    "TPU_WORKER_ID":"0",
    "KERAS_BACKEND":"jax",
}
POLICY={
    512:("segmented",256),
    768:("segmented",256),
    1024:("segmented-query-chunk",256),
}

def run_stage(name,cmd,env,log):
    print(f"=== {name} ===",flush=True)
    started=time.time()
    with open(log,"w",encoding="utf-8") as fh:
        proc=subprocess.run(cmd,env=env,stdout=fh,stderr=subprocess.STDOUT,text=True)
    if proc.returncode!=0:
        print(Path(log).read_text(errors="replace")[-8000:],file=sys.stderr)
        raise SystemExit(f"{name} failed rc={proc.returncode}")
    print(Path(log).read_text(errors="replace")[-4000:],flush=True)
    return time.time()-started

def main():
    ap=argparse.ArgumentParser(description="Mage-Flow TPU v5e-8 production inference runner")
    ap.add_argument("--topology",choices=("1x8","2x4","4x2"),default="4x2")
    ap.add_argument("--resolution",type=int,choices=(512,768,1024),default=512)
    ap.add_argument("--seeds",default=None)
    ap.add_argument("--steps",type=int,default=4)
    ap.add_argument("--attention",choices=("auto","masked-global","segmented","segmented-query-chunk"),default="auto")
    ap.add_argument("--query-chunk",type=int,default=256)
    for k in ("model-root","runtime-root","text-checkpoint","transformer-checkpoint","vae-checkpoint","vae-manifest","basis-dim16","basis-dim56","output"):
        ap.add_argument("--"+k,required=True)
    ap.add_argument("--runtime-site", default=None)
    ap.add_argument(
        "--source-root",
        default=str(PROJECT_ROOT / "runtime/final/source/mage_flow_keras"),
    )
    args=ap.parse_args()
    if args.steps!=4: raise SystemExit("only qualified --steps 4 is supported")
    replicas=int(args.topology.split("x")[0])
    default_seeds=",".join(str(42+i) for i in range(replicas))
    seeds=args.seeds or default_seeds
    seed_list=[x for x in seeds.split(",") if x.strip()]
    if len(seed_list)!=replicas: raise SystemExit(f"{args.topology} requires {replicas} seeds")
    mode,auto_chunk=POLICY[args.resolution]
    if args.attention!="auto": mode=args.attention
    chunk=args.query_chunk if args.attention!="auto" else auto_chunk
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    logs=out/"logs"; logs.mkdir(exist_ok=True)
    text_out=out/"text"; tf_out=out/"transformer"; vae_out=out/"vae"
    runtime_root = Path(args.runtime_root).resolve()
    source_root = Path(args.source_root).resolve()
    runtime_site = Path(args.runtime_site).resolve() if args.runtime_site else None
    for label, path in (("runtime-root", runtime_root), ("source-root", source_root)):
        if not path.exists():
            raise SystemExit(f"{label} does not exist: {path}")
    if runtime_site is not None and not runtime_site.exists():
        raise SystemExit(f"runtime-site does not exist: {runtime_site}")
    env=os.environ.copy()
    env.update(TPU_ENV)
    pythonpath = []
    if runtime_site is not None:
        pythonpath.append(str(runtime_site))
    pythonpath.extend([str(runtime_root), str(source_root.parent), env.get("PYTHONPATH","")])
    env["PYTHONPATH"]=":".join(x for x in pythonpath if x)
    py=sys.executable
    timings={}
    timings["text"]=run_stage("TEXT",[
        py,str(SCRIPT_DIR/"06_tpu_stage_text.py"),
        "--runtime-root",str(runtime_root),
        "--model-root",args.model_root,
        "--checkpoint",args.text_checkpoint,
        "--output",str(text_out),
    ],env,logs/"text.log")
    timings["transformer"]=run_stage("TRANSFORMER",[
        py,str(SCRIPT_DIR/"06_tpu_stage_transformer.py"),
        "--runtime-root",str(runtime_root),
        "--source-root",str(source_root),
        "--checkpoint",args.transformer_checkpoint,
        "--basis-dim16",args.basis_dim16,
        "--basis-dim56",args.basis_dim56,
        "--conditioning",str(text_out/"C0_text_conditioning_f32.npy"),
        "--output",str(tf_out),
        "--topology",args.topology,
        "--resolution",str(args.resolution),
        "--steps",str(args.steps),
        "--seeds",seeds,
        "--attention-mode",mode,
        "--query-chunk",str(chunk),
    ],env,logs/"transformer.log")
    timings["vae"]=run_stage("VAE",[
        py,str(SCRIPT_DIR/"06_tpu_stage_vae.py"),
        "--runtime-root",str(runtime_root),
        "--checkpoint",args.vae_checkpoint,
        "--manifest",args.vae_manifest,
        "--input-dir",str(tf_out),
        "--output",str(vae_out),
        "--topology",args.topology,
        "--resolution",str(args.resolution),
        "--seeds",seeds,
    ],env,logs/"vae.log")
    tf=json.loads((tf_out/"transformer_stage_summary.json").read_text())
    vae=json.loads((vae_out/"vae_stage_summary.json").read_text())
    text_summary=json.loads((text_out/"text_stage_summary.json").read_text())
    summary={
        "status":"PASS","runner":"06_run_tpu_inference.py",
        "topology":args.topology,"resolution":[args.resolution,args.resolution],
        "steps":args.steps,"seeds":[int(x) for x in seed_list],
        "attention_mode":mode,
        "query_chunk":chunk if mode=="segmented-query-chunk" else None,
        "policy_source":"production-auto" if args.attention=="auto" else "explicit-override",
        "stage_wall_s":timings,
        "text":text_summary,
        "transformer":tf,
        "vae":vae,
    }
    (out/"production_run_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    main()
