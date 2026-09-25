#!/usr/bin/env python3
import argparse,json,time
from pathlib import Path
import numpy as np
import jax

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runtime-root",required=True)
    ap.add_argument("--model-root",required=True)
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    import sys
    sys.path.insert(0,args.runtime_root)
    from experiment import text_encoder_driver as ted
    from experiment.runtime_contract import PROMPT
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    assert jax.default_backend()=="tpu" and len(jax.devices())==8
    t=time.time()
    handle=ted.restore_text_encoder(
        text_encoder_checkpoint=args.checkpoint,
        model_root=args.model_root,
        execute_compute=True,
    )
    result=ted.encode_text(handle,[PROMPT],hooks=None,execute_compute=True)
    cond=np.asarray(jax.device_get(result["outputs"][0]["tensor"])).astype(np.float32)
    assert cond.shape==(1,19,2560) and np.isfinite(cond).all()
    np.save(out/"C0_text_conditioning_f32.npy",cond)
    summary={"status":"PASS","backend":jax.default_backend(),"device_count":len(jax.devices()),
             "shape":list(cond.shape),"dtype":str(cond.dtype),"elapsed_s":time.time()-t}
    (out/"text_stage_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)
if __name__=="__main__": main()
