# dnh: do-no-harm latent steering (code for the IROS draft)

Frozen LeRobot Diffusion Policy on Push-T, steered at inference time by a probabilistic
world model, with steering strength + abstention threshold certified by Learn-then-Test.
Paper draft: [`../proposal/iros/main.tex`](../proposal/iros/main.tex).

## Run on Colab (A100)
[Open the runner in Colab](https://colab.research.google.com/github/carlo-scr/dnh-latent-steering/blob/main/notebooks/colab_runner.ipynb), select the A100 runtime, and run top to bottom.
The setup cell clones (or pulls) this repo; all outputs go to `MyDrive/dnh_runs`.

## Pipeline
| Step | Script | What it gives you |
|---|---|---|
| 0 | `00_smoke_test.py` | exact snapshot/restore, perturbations, K-batched sampling, paired re-run determinism |
| 1 | `01_collect.py` | base-policy rollouts (nominal → success-WM; perturbed failures → failure-WM) |
| 2 | `02_train_wm.py` | stand-in probabilistic WMs (ensemble, Gaussian NLL) |
| 3 | `03_rq1_ranking.py` | **week-1 go/no-go**: executes all K candidates from branch states; AUROC + steering gain per scorer |
| 4 | `04_calibrate_ltt.py` | Algorithm 1: paired harm + false-abstention, LTT on a cal split, empirical risk on a test split |
| 5 | `05_eval_shift.py` | success / abstain / stall fraction under 6 perturbation families at the certified setting |

## Design notes
- **λ=0 is the base policy, exactly.** Candidates are i.i.d. policy samples and λ=0 executes candidate 0. Per-episode seeded noise means paired base/steered runs share random numbers until their states diverge.
- **The stand-in WM runs on low-dim state** (agent xy, block xy, sin/cos θ), not on Cosmos image latents. It exists so the pipeline runs today. To plug in the lab's Cosmos-latent WM (Ward et al.), implement `Scorer.score(ctx, chunks) -> [B,K]` in `dnh/world_model.py`. The context would then need image latents; extend `Context` and `rollout._context`.
- **Sampler:** the default is the checkpoint's original 100-step DDPM, whose denoising is stochastic. `--ddim N` is faster and deterministic given the initial noise, which noise-space steering needs, but it slightly changes the base policy. Use the same setting for every step of an experiment.
- **Checkpoint:** `lerobot/diffusion_pusht` predates lerobot 0.4 processors. `migrate_checkpoint` converts it once (published eval: 65.4 % success).
- **Why not Isaac on Colab:** Isaac Sim needs an RTX GPU (RT cores) for rendering, and the A100 has none, so camera-based policies can't run there. The next task (robomimic Transport / LIBERO) uses MuJoCo with `MUJOCO_GL=egl`, which works headless on A100.

## Local (CPU) sanity run
```bash
uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e .
pytest -q tests
python scripts/00_smoke_test.py --ckpt runs/dp_pusht_migrated --device cpu --n 2 --K 4 --ddim 3 --max-replans 5
```
