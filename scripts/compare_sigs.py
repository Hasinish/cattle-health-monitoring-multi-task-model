import hashlib
import os

def md5(fname):
    with open(fname, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

gra_sig = "cattle_thesis_p3_latex/images/sig_gra.jpg"
print(f"sig_gra.jpg md5: {md5(gra_sig)} (size: {os.path.getsize(gra_sig)} bytes)")

sig_dir = "scratch/extracted_signatures"
for f in os.listdir(sig_dir):
    fpath = os.path.join(sig_dir, f)
    if os.path.isfile(fpath):
        print(f"{f}: md5={md5(fpath)} (size: {os.path.getsize(fpath)} bytes)")

hod_dir = "scratch/hod_signatures"
for f in os.listdir(hod_dir):
    fpath = os.path.join(hod_dir, f)
    if os.path.isfile(fpath):
        print(f"HOD/{f}: md5={md5(fpath)} (size: {os.path.getsize(fpath)} bytes)")
