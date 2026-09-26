from PIL import Image
import os

sig_dir = "scratch/extracted_signatures"
for fname in os.listdir(sig_dir):
    fpath = os.path.join(sig_dir, fname)
    if os.path.isfile(fpath):
        im = Image.open(fpath)
        print(f"{fname}: format={im.format}, size={im.size}, mode={im.mode}")
