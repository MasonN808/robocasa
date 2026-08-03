import json,glob,numpy as np
from training.bc_task_vlm.live_sim_eval import SimSession
D="/work/umass/shlomo_umass/dbenhamougol_umass/data"
ROOT=f"{D}/structured_random_raw/data_generation/task_level/data_structured_random/image/20260706T174208"
for task,pat in (("arrange_bread_bowl","toaster"),("sweeten_coffee","coffee")):
    p=sorted(glob.glob(f"{ROOT}/{task}/traj_*/original_trajectory.json"))[0]
    traj=json.load(open(p))
    s=SimSession(composite_task=traj["composite_task"],sample_trajectory=traj,
                 layout=11,style=34,seed=42,gl_backend="egl",render_size=256)
    fx=s.executor.runner._fixtures
    pos={k:np.asarray(v.pos[:2],dtype=float) for k,v in fx.items() if getattr(v,"pos",None) is not None}
    app=[k for k in pos if pat in k.lower()]
    cnt=[k for k in pos if any(t in k.lower() for t in ("counter","dining","island"))]
    print(f"\n### {task}")
    for a in app:
        d=sorted((float(np.linalg.norm(pos[a]-pos[c])),c) for c in cnt)
        print(f"  {a}:")
        for dist,c in d[:3]: print(f"     {dist:6.2f} m  ->  {c}")
