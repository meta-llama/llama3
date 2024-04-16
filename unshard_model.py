from typing import List, Optional
import torch
import fire
from pathlib import Path
import re

def main(
    ckpt_dir: str,
):
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))

    assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
    cps = []
    check_name=re.compile("[^.]*[.][0-9]*[.]pth$")
    for cp in checkpoints:
        assert check_name.match(cp.name), f"found {cp.name} which does not match expected name format <name>.<digits>.pth"
        cps.append(torch.load(cp, map_location="cpu", mmap=True))
    print(f"Combining {''.join([x.name+', ' for x in checkpoints])} into combined.pth")
    combined_cp={}
    for key in cps[0].keys():
        if not torch.allclose(cps[0][key], cps[1][key]):
            values = (cps[0][key], cps[1][key], cps[2][key], cps[3][key])
            if "wo" in key or "w2" in key:
                # Concat on dim=1 for "wo" and "w2".
                combined_cp[key] = torch.cat(values, dim=1)
            else:
                # Concat on dim=0 for everything else.
                combined_cp[key] = torch.cat(values, dim=0)
        else:
            # Do not duplicate layers shared between each checkpoint.
            combined_cp[key] = cps[0][key]
    torch.save(combined_cp, Path(ckpt_dir) / "model.pth")


if __name__ == "__main__":
    fire.Fire(main)
