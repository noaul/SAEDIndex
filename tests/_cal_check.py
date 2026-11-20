import glob, sys, collections, numpy as np
from PIL import Image
from saedindex.calibration import detect_scale_bar, read_bar_label
from saedindex.dmfile import read_dm
files = sorted(glob.glob(r"F:\E D\**\*.jpg", recursive=True))
stats = collections.Counter(); errs=[]
for p in files:
    try:
        _, s, u, _ = read_dm(p[:-4]+".dm3", load_data=False)
    except Exception: continue
    if not u.startswith("1/"): continue
    g = np.asarray(Image.open(p).convert("L"))
    bar = detect_scale_bar(g)
    if not bar: stats["nobar"]+=1; print("NOBAR", p[-40:]); continue
    lab = read_bar_label(g, bar)
    diff = u.startswith("1/")
    if not lab["value"]: stats["noval"]+=1; print("NOVAL", p[-40:]); continue
    truth = 1/s if diff else 1/s  # px per unit
    est = bar["pixels"]/lab["value"]
    err = (est/truth-1)*100
    unit_ok = lab["unit"].startswith("1/") == diff
    stats["unit_ok" if unit_ok else "unit_bad"]+=1
    if abs(err) < 3: stats["ok"]+=1; errs.append(err)
    else:
        stats["bad"]+=1
        if stats["bad"]<25: print("BAD", p[-45:], bar["pixels"], bar["outlined"], lab["text"], round(lab["confidence"],2), "true value", round(bar["pixels"]*s,3), u)
print(stats, "mean|err|", np.mean(np.abs(errs)), "max", np.max(np.abs(errs)))
