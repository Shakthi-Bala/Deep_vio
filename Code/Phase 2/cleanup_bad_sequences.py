"""Post-generation cleanup: remove sequences with blur < 30."""
import cv2, os, shutil
import numpy as np

output_dir = 'output'
removed = 0
kept = 0

for split in ['train', 'val', 'test']:
    split_dir = os.path.join(output_dir, split)
    if not os.path.exists(split_dir):
        continue
    
    seqs = sorted(os.listdir(split_dir))
    for seq in seqs:
        img_dir = os.path.join(split_dir, seq, 'images')
        if not os.path.exists(img_dir):
            continue
        
        # Sample 3 frames
        blur_scores = []
        for frame in ['00005.png', '00050.png', '00100.png']:
            fpath = os.path.join(img_dir, frame)
            if not os.path.exists(fpath):
                continue
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                blur_scores.append(cv2.Laplacian(img, cv2.CV_64F).var())
        
        if not blur_scores:
            continue
        
        avg_blur = np.mean(blur_scores)
        if avg_blur < 30:
            shutil.rmtree(os.path.join(split_dir, seq))
            removed += 1
            print(f"  REMOVED {split}/{seq} (avg_blur={avg_blur:.1f})")
        else:
            kept += 1

print(f"
Done. Kept: {kept}, Removed: {removed}")
print(f"Remaining sequences per split:")
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(output_dir, split)
    if os.path.exists(split_dir):
        n = len(os.listdir(split_dir))
        print(f"  {split}: {n}")

# Renumber sequences to be contiguous
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(output_dir, split)
    if not os.path.exists(split_dir):
        continue
    seqs = sorted(os.listdir(split_dir))
    for i, seq in enumerate(seqs, 1):
        new_name = f"seq_{i:03d}"
        if seq != new_name:
            os.rename(os.path.join(split_dir, seq), os.path.join(split_dir, new_name))

print("
Sequences renumbered.")
