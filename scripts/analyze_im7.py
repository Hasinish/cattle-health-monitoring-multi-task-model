from PIL import Image
import numpy as np

im = Image.open("scratch/extracted_signatures/Final_Report - ABRAR MAHIR ROHAN.pdf_p4_img0_Im7.jpg")
arr = np.array(im)
print("Min value:", arr.min(), "Max value:", arr.max(), "Mean value:", arr.mean())

# If there are any non-white pixels, let's find their coordinates and values
non_white = np.where(arr < 250)
print(f"Number of non-white (<250) pixels: {len(non_white[0])}")
if len(non_white[0]) > 0:
    print("Min pixel val:", arr[non_white].min())
    # Save enhanced contrast
    contrast = 255 - ((255 - arr) * 10)
    contrast = np.clip(contrast, 0, 255).astype(np.uint8)
    Image.fromarray(contrast).save("scratch/im7_contrast.png")
    print("Saved scratch/im7_contrast.png")
