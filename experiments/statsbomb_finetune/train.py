"""Local Qwen2-VL probability-head/QLoRA pilot. No paid API or GPU rental code."""
import argparse
import json
import math
from pathlib import Path
import random
import time

from runtime import OUT, MODEL_ID, read, save, preflight


def load_split(split, protocol):
    """Training calls this only for train/validation; test is parsed solely by evaluate.py."""
    from common import sha, verify_image
    path = OUT/f"{split}.jsonl"
    if sha(path) != protocol["jsonl_sha256"][split]:
        raise ValueError("Split manifest changed: " + split)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    allowed = {"shot_id","match_id","image","image_sha256","label"}
    for row in records:
        if set(row) != allowed or type(row["label"]) is not int or row["label"] not in (0,1):
            raise ValueError("Invalid supervised example or forbidden model metadata")
        verify_image(row,OUT)
    return records


def check_training_protocol():
    from common import MODEL_ID as PREPARED_MODEL_ID, PROMPT, sha
    if PREPARED_MODEL_ID != MODEL_ID:
        raise ValueError("Runtime and prepared model IDs differ")
    protocol = read(OUT/"protocol.json")
    if protocol["model_id"] != MODEL_ID or protocol["prompt"] != PROMPT:
        raise ValueError("Frozen model/prompt changed")
    for name, key in (("selected.json","selected_sha256"),("match_plan.json","match_plan_sha256"),("source_audit.json","source_audit_sha256")):
        if sha(OUT/name) != protocol[key]:
            raise ValueError("Frozen data changed: " + name)
    for filename, expected in protocol["code_sha256"].items():
        if sha(Path(__file__).with_name(filename)) != expected:
            raise ValueError("Preparation code changed; use a new versioned experiment")
    plan = read(OUT/"match_plan.json")["splits"]
    sets = {k:set(v) for k,v in plan.items()}
    if any(sets[a]&sets[b] for a,b in (("train","validation"),("train","test"),("validation","test"))):
        raise ValueError("Overlapping match assignments")
    return protocol


def build_model(arm, prior, allow_download=False, revision=None):
    from common import SEED
    import torch
    from transformers import AutoConfig, AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    access = {"local_files_only":not allow_download,"cache_dir":str(OUT/"model_cache")}
    config = AutoConfig.from_pretrained(MODEL_ID, revision=revision, **access)
    revision = revision or getattr(config,"_commit_hash",None)
    if not revision:
        raise ValueError("A pinned pretrained-model revision is required")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=revision,
        min_pixels=256*28*28,max_pixels=256*28*28,**access)
    quant = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
    backbone = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID,revision=revision,
        quantization_config=quant,device_map={"":"cuda:0"},torch_dtype=dtype,
        attn_implementation="sdpa",**access)
    backbone.config.use_cache = False
    backbone = prepare_model_for_kbit_training(backbone,use_gradient_checkpointing=arm=="lora",
        gradient_checkpointing_kwargs={"use_reentrant":False})
    if arm == "lora":
        targets = [name for name,_ in backbone.named_modules()
                   if name.endswith(("q_proj","v_proj")) and "visual" not in name]
        if not targets:
            raise ValueError("No text attention modules matched")
        backbone = get_peft_model(backbone,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,
            target_modules=targets,bias="none",task_type="CAUSAL_LM"))
    else:
        backbone.requires_grad_(False)
        backbone.eval()
    if any(p.requires_grad for name,p in backbone.named_parameters() if "visual" in name):
        raise ValueError("Vision encoder must remain frozen")
    cfg = getattr(backbone.config,"text_config",backbone.config)
    hidden = cfg.hidden_size
    torch.manual_seed(SEED+2)  # Identical head initialization in the two arms.
    torch.cuda.manual_seed_all(SEED+2)
    head = torch.nn.Linear(hidden,1,device="cuda",dtype=torch.float32)
    torch.nn.init.normal_(head.weight,std=.001)
    torch.nn.init.constant_(head.bias,math.log(prior/(1-prior)))
    return backbone,head,processor,revision


def make_inputs(processor, row):
    from common import PROMPT
    from PIL import Image
    import torch
    messages = [{"role":"user","content":[{"type":"image"},{"type":"text","text":PROMPT}]}]
    text = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    with Image.open(OUT/row["image"]) as source:
        image = source.convert("RGB")
        inputs = processor(text=[text],images=[image],padding=True,return_tensors="pt")
    # Outcome label is deliberately not consulted here and is never tokenized into the prompt.
    return {k:v.to("cuda") if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}


def forward(backbone,head,inputs,frozen=False):
    import torch
    if frozen:
        with torch.no_grad():
            result = backbone(**inputs,output_hidden_states=True,return_dict=True,use_cache=False)
    else:
        result = backbone(**inputs,output_hidden_states=True,return_dict=True,use_cache=False)
    hidden = result.hidden_states[-1]
    positions = (inputs["attention_mask"]*torch.arange(hidden.shape[1],device=hidden.device)).max(dim=1).values
    features = hidden[torch.arange(hidden.shape[0],device=hidden.device),positions].float()
    return head(features).squeeze(-1)


def probabilities(backbone,head,processor,rows,deadline=None):
    import torch
    backbone.eval()
    head.eval()
    values = []
    with torch.no_grad():
        for row in rows:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("Local wall-clock limit reached during validation")
            logits = forward(backbone,head,make_inputs(processor,row),frozen=True)
            values.append(float(torch.sigmoid(logits).item()))
    return values


def train(arm, smoke=False, allow_download=False, status_path=None):
    status = preflight()
    save(status_path or OUT/"training_preflight.json",status)
    if status["status"] != "runtime_ready_weights_not_yet_checked":
        print(json.dumps(status,indent=2),flush=True)
        return 2
    import torch
    import numpy as np
    from common import SEED, sha, metrics
    protocol = check_training_protocol()
    if (OUT/"test_evaluation.json").exists():
        raise ValueError("Test already evaluated; no further tuning in this version")
    data = {s:load_split(s,protocol) for s in ("train","validation")}
    if {r["match_id"] for r in data["train"]}&{r["match_id"] for r in data["validation"]}:
        raise ValueError("Training and validation match leakage")
    destination = OUT/("smoke_"+arm if smoke else arm)
    if destination.exists():
        raise ValueError("Run directory exists; preserve it and use a new version for another attempt")
    destination.mkdir()
    code = {p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name("common.py"),Path(__file__).with_name("runtime.py"))}
    manifest = {"arm":arm,"smoke_only":smoke,"protocol_sha256":sha(OUT/"protocol.json"),"code_sha256":code,
        "model_id":MODEL_ID,"runtime":status,"seed":SEED,"test_examples_loaded":0,
        "paid_api_calls":0,"started_unix":time.time(),"status":"starting"}
    save(destination/"manifest.json",manifest)
    started = time.monotonic()
    deadline = started+protocol["training"]["max_seconds_per_arm"]
    random.seed(SEED)
    np.random.seed(SEED)
    prior = sum(r["label"] for r in data["train"])/len(data["train"])
    try:
        other_arm = "head" if arm=="lora" else "lora"
        other = OUT/other_arm/"manifest.json"
        revision = read(other).get("base_revision") if other_arm in {p.name for p in OUT.iterdir()} and other.is_file() else None
        backbone,head,processor,revision = build_model(arm,prior,allow_download,revision)
        trainable = [p for p in backbone.parameters() if p.requires_grad]
        manifest.update(base_revision=revision,trainable_backbone_parameters=sum(p.numel() for p in trainable),
                        trainable_head_parameters=sum(p.numel() for p in head.parameters()),status="training")
        save(destination/"manifest.json",manifest)
        groups = [{"params":list(head.parameters()),"lr":protocol["training"]["head_lr"]}]
        if trainable:
            groups.append({"params":trainable,"lr":protocol["training"]["adapter_lr"]})
        optimizer = torch.optim.AdamW(groups,weight_decay=protocol["training"]["weight_decay"])
        loss_fn = torch.nn.BCEWithLogitsLoss()
        history, best = [], float("inf")
        accum = protocol["training"]["gradient_accumulation"]
        for epoch in range(1,protocol["training"]["epochs"]+1):
            head.train()
            backbone.train() if arm=="lora" else backbone.eval()
            order = list(data["train"])
            random.Random(SEED+epoch).shuffle(order)
            if smoke:
                order = order[:1]
            losses = []
            for first in range(0,len(order),accum):
                if time.monotonic() > deadline:
                    raise TimeoutError("Local wall-clock limit reached during training")
                group = order[first:first+accum]
                optimizer.zero_grad(set_to_none=True)
                for row in group:
                    logits = forward(backbone,head,make_inputs(processor,row),frozen=arm=="head")
                    label = torch.tensor([row["label"]],device="cuda",dtype=torch.float32)
                    loss = loss_fn(logits.float(),label)
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite BCE loss")
                    (loss/len(group)).backward()
                    losses.append(float(loss.detach().item()))
                torch.nn.utils.clip_grad_norm_(list(head.parameters())+trainable,1.)
                optimizer.step()
                if first%80==0:
                    print(f"{arm}: epoch {epoch}, {first+len(group)}/{len(order)} shots, mean BCE {sum(losses)/len(losses):.4f}",flush=True)
            if smoke:
                manifest.update(status="smoke_passed",train_micro_batches=1,
                                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started)
                save(destination/"manifest.json",manifest)
                print("One-batch forward/backward smoke test passed; no trained model result claimed",flush=True)
                return 0
            p = probabilities(backbone,head,processor,data["validation"],deadline)
            scores = metrics([r["label"] for r in data["validation"]],p)
            history.append({"epoch":epoch,"training_bce":sum(losses)/len(losses),"validation":scores,
                            "elapsed_seconds":time.monotonic()-started})
            save(destination/"history.json",history)
            if scores["brier"] < best:
                best = scores["brier"]
                checkpoint = destination/"best"
                checkpoint.mkdir(exist_ok=True)
                torch.save(head.state_dict(),checkpoint/"head.pt")
                if arm=="lora":
                    backbone.save_pretrained(checkpoint/"adapter",safe_serialization=True)
                processor.save_pretrained(checkpoint/"processor")
                save(checkpoint/"selection.json",{"epoch":epoch,"validation_brier":best,"base_revision":revision,
                    "protocol_sha256":sha(OUT/"protocol.json"),"head_sha256":sha(checkpoint/"head.pt"),
                    "adapter_sha256":{str(p.relative_to(checkpoint)):sha(p) for p in (checkpoint/"adapter").rglob("*") if p.is_file()} if arm=="lora" else {}})
            print(f"{arm}: complete epoch {epoch}, validation Brier {scores['brier']:.5f}",flush=True)
        manifest.update(status="completed",epochs_completed=len(history),best_validation_brier=best,
                        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started)
        save(destination/"manifest.json",manifest)
        return 0
    except Exception as error:
        manifest.update(status="blocked_or_failed",error_type=type(error).__name__,elapsed_seconds=time.monotonic()-started)
        # Save a bounded diagnostic; no credentials are ever read by this program.
        manifest["reason"] = str(error)[:1000]
        save(destination/"manifest.json",manifest)
        print(json.dumps({"status":manifest["status"],"error_type":manifest["error_type"],"reason":manifest["reason"]}),flush=True)
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm",choices=["head","lora"],default="lora")
    parser.add_argument("--execute",action="store_true",help="Run local training; never submits paid work")
    parser.add_argument("--smoke",action="store_true",help="One forward/backward batch, no accuracy claim")
    parser.add_argument("--allow-download",action="store_true",help="Allow downloading public pretrained weights; default requires cached weights")
    parser.add_argument("--status-file",type=Path,help="Write runtime diagnostics to this path instead of the project status file")
    args = parser.parse_args()
    if args.execute:
        raise SystemExit(train(args.arm,args.smoke,args.allow_download,args.status_file))
    status = preflight()
    save(args.status_file or OUT/"training_preflight.json",status)
    print(json.dumps(status,indent=2),flush=True)


if __name__ == "__main__":
    main()
