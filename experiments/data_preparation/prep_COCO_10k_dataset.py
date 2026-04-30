import json
from pathlib import Path

COCO_ROOT   = '/scratch/kt68/coco_dataset'
TRAIN_TXT   = '/scratch/kt68/training_datasets/coco_10k/train.txt'
LABELS_ROOT = Path(COCO_ROOT) / 'labels'

COCO91_TO_80 = {
    1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9,
    11:10,13:11,14:12,15:13,16:14,17:15,18:16,19:17,20:18,21:19,
    22:20,23:21,24:22,25:23,27:24,28:25,31:26,32:27,33:28,34:29,
    35:30,36:31,37:32,38:33,39:34,40:35,41:36,42:37,43:38,44:39,
    46:40,47:41,48:42,49:43,50:44,51:45,52:46,53:47,54:48,55:49,
    56:50,57:51,58:52,59:53,60:54,61:55,62:56,63:57,64:58,65:59,
    67:60,70:61,72:62,73:63,74:64,75:65,76:66,77:67,78:68,79:69,
    80:70,81:71,82:72,84:73,85:74,86:75,87:76,88:77,89:78,90:79,
}

def convert_split(json_path, split_name, allowed_stems=None):
    print(f'Converting {split_name}...')
    with open(json_path) as f:
        data = json.load(f)

    # Filter images to only those we need
    images = {
        img['id']: img for img in data['images']
        if allowed_stems is None
        or Path(img['file_name']).stem in allowed_stems
    }
    print(f'  Processing {len(images):,} images')

    ann_by_image = {}
    for ann in data['annotations']:
        if ann['image_id'] in images:
            ann_by_image.setdefault(ann['image_id'], []).append(ann)

    out_dir = LABELS_ROOT / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_id, img_info in images.items():
        anns = ann_by_image.get(img_id, [])
        w, h = img_info['width'], img_info['height']
        stem = Path(img_info['file_name']).stem

        lines = []
        for ann in anns:
            cls80 = COCO91_TO_80.get(ann.get('category_id'))
            if cls80 is None or ann.get('iscrowd', 0):
                continue
            bbox = ann.get('bbox')
            if not bbox:
                continue
            x, y, bw, bh = bbox
            cx = max(0.0, min(1.0, (x + bw/2) / w))
            cy = max(0.0, min(1.0, (y + bh/2) / h))
            nw = max(0.0, min(1.0, bw / w))
            nh = max(0.0, min(1.0, bh / h))
            if nw > 0 and nh > 0:
                lines.append(f'{cls80} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}')

        with open(out_dir / f'{stem}.txt', 'w') as f:
            f.write('\n'.join(lines) + '\n' if lines else '')

    print(f'  Done: {len(images):,} label files → {out_dir}')

# Read the 10k stems from your existing train.txt
with open(TRAIN_TXT) as f:
    train_stems = {Path(line.strip()).stem for line in f if line.strip()}
print(f'Loaded {len(train_stems):,} stems from train.txt')

# Convert only the 10k training images (fast — filters before writing)
convert_split(
    f'{COCO_ROOT}/annotations/instances_train2017.json',
    'train2017',
    allowed_stems=train_stems,
)

# Always convert all val2017 — only 5k images, needed for evaluation
convert_split(
    f'{COCO_ROOT}/annotations/instances_val2017.json',
    'val2017',
    allowed_stems=None,
)

print('All done.')