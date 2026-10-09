"""Train the stand-in probabilistic WMs: success-WM on nominal successes, failure-WM on
failed perturbed rollouts (all files matching --failure-glob).
    python scripts/02_train_wm.py --root $ROOT
"""
from pathlib import Path

from dnh.experiment import base_parser, load_pickle
from dnh.world_model import save_wm, train_wm

ap = base_parser(__doc__)
ap.add_argument("--H", type=int, default=4)
ap.add_argument("--members", type=int, default=5)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--failure-glob", default="rollouts_*[0-9].pkl")
args = ap.parse_args()
root = Path(args.root)

allnom = load_pickle(root / "data" / "rollouts_nominal.pkl")
nom = [e for e in allnom if e["success"]]
if len(nom) < 10:
    print(f"WARNING: only {len(nom)} nominal successes; training success-WM on all {len(allnom)} nominal episodes")
    nom = allnom
print(f"success-WM: {len(nom)} nominal episodes")
save_wm(train_wm(nom, args.H, args.members, args.epochs, device=args.device), root / "wm_success.pt")

fails = []
for f in sorted((root / "data").glob(args.failure_glob)):
    fails += [e for e in load_pickle(f) if not e["success"]]
if fails:
    print(f"failure-WM: {len(fails)} failed perturbed episodes")
    save_wm(train_wm(fails, args.H, args.members, args.epochs, device=args.device), root / "wm_failure.pt")
else:
    print("no perturbed rollouts found -> skipping failure-WM (contrast scorer unavailable)")
