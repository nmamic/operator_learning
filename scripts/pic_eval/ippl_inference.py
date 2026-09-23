import os, sys
from pathlib import Path
import time
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch, cupy as cp, h5py
import torch.distributed as dist
from torch.utils.dlpack import from_dlpack
from training.train_fno import FourierNeuralOperator
from operator_learning.data.pic_dataset import normalize_per_sample, normalize_per_sample_distributed
from operator_learning.utils.misc import readConfig

_S = {}

def init(rank, size, device_id, params):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(size)
    os.environ["LOCAL_RANK"] = str(device_id)

    cfg = readConfig(os.environ["IPPL_CONFIG"])

    if "trainDir" in cfg.train:
        FourierNeuralOperator.TRAIN_DIR = cfg.train.trainDir

    strategy = None if size == 1 else {
        "gpus_per_node": cfg.parallel_strategy.get("gpus_per_node", 4),
        "ddp": True, "tp": True, "tp_size": size,
    }
    model = FourierNeuralOperator(
        checkpoint=cfg.train.checkpoint, eval_only=True, device="cuda",
        data_class="pic", model_dtype=torch.float32, fno_dtype=torch.float32,
        parallel_strategy=strategy)

    assert torch.cuda.current_device() == device_id, (
        f"rank {rank}: torch on device {torch.cuda.current_device()}, "
        f"Kokkos on {device_id}")
    
    cp.cuda.Device(device_id).use()

    with h5py.File(cfg.data.dataFile, "r") as d:
        out_mean = d["infos"]["output_mean"][()]
        out_std  = d["infos"]["output_std"][()]
    L = cp.asarray(params["L"])
    _S.update(model=model, dim=int(params["dim"]),
              tp_mesh=model.tp_mesh if model.TP_enabled else None,
              tp_enabled=model.TP_enabled,
              out_mean=out_mean,
              out_std=out_std,
              N=int(params["totalP"]), Q=float(params["q_per_particle"]),
              scale=float(params["Q_total"]) / float(cp.prod(L) ** (2/3)))

def infer(R_cap, E_cap, nlocal):
    t0 = time.perf_counter()
    R = from_dlpack(R_cap).T # (dim, N_local) float64, this is transposed from IPPL
                             # which saves as (N_local, dim)
    E = from_dlpack(E_cap).T #the write target
    if not _S.get("nlocal_printed", False):
        print(f"[rank {os.environ['RANK']}] nlocal = {nlocal}", flush=True)
        _S["nlocal_printed"] = True
    inputs = cp.from_dlpack(R)[None, :, :].copy()
    if _S["tp_enabled"]:
        lo = inputs.min(axis=2, keepdims=True)
        hi = inputs.max(axis=2, keepdims=True)
        inputs = normalize_per_sample_distributed(inputs, _S["tp_mesh"])
    else:
        for ch in range(_S["dim"]):
            inputs[:, ch, :] = normalize_per_sample(inputs[:, ch, :])

    pred = _S["model"](inputs).squeeze(0)

    # taken directly from PicND 3D branch
    for ch in range(_S["dim"]):
        pred[ch] = pred[ch] * _S["out_std"][ch] + _S["out_mean"][ch]
    pred[:, :] = pred[:, :] * _S["scale"]
    for ch in range(_S["dim"]):
        s = pred[ch].sum(dtype=cp.float64)
        if _S["tp_enabled"]:
            t = torch.from_dlpack(s.toDlpack())
            dist.all_reduce(t, dist.ReduceOp.SUM, group=_S["tp_mesh"].get_group())
        pred[ch] -= s / _S["N"]  

    E.copy_(torch.from_dlpack(pred.toDlpack()).to(torch.float64))
    torch.cuda.synchronize()

def finalize():
    if dist.is_initialized():
        dist.destroy_process_group()
    _S.clear()