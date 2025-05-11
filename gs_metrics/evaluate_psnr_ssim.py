import os
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

original_dir = "/home/dhiren/Agisoft_Dataset"
rendered_dir = "/home/dhiren/gaussian-splatting/output/551368e8-e/test/ours_30000/renders"
output_table = []

# Match files by order
original_files = sorted([f for f in os.listdir(original_dir) if f.endswith(".png")])
rendered_files = sorted([f for f in os.listdir(rendered_dir) if f.endswith(".png")])

for orig_file, rend_file in zip(original_files, rendered_files):
    orig_path = os.path.join(original_dir, orig_file)
    rend_path = os.path.join(rendered_dir, rend_file)

    img1 = cv2.imread(orig_path)
    img2 = cv2.imread(rend_path)

    if img1 is None or img2 is None:
        continue

    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))  # match dimensions
    psnr = peak_signal_noise_ratio(img1, img2, data_range=255)
    ssim = structural_similarity(img1, img2, channel_axis=2)

    output_table.append((orig_file, rend_file, round(psnr, 2), round(ssim, 4)))

print("Original\tRendered\tPSNR\tSSIM")
for row in output_table:
    print("\t".join(map(str, row)))

