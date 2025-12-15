# KFS-Bench

This repository provides KFS-Bench dataset and evaluation code from the paper:
```text
Z. Li, K. Ishida, S. Yamazaki, X. Ji and J. Liu, "KFS-Bench: Comprehensive Evaluation of Key Frame Sampling in Long Video Understanding", IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), Tucson, Arizona, 2026.
```

KFS-Bench is the first benchmark for long video QA with multi-scene annotations, enabling direct evaluation of key frame sampling.


## Ground Truth Data Format

The CSV files contain the following columns:
- `id`: Video identifier
- `segment_id`: Segment identifier
- `scene_id`: Scene identifier (for grouping related segments)
- `start`: Start timestamp
- `end`: End timestamp

## Evaluation

```bash
python eval.py --anno_file lvb_anno.csv --result_file your_results.json 
```

You can perform the same evaluations as those conducted in the experiments described in the paper.
```bash
python eval.py --anno_file lvb_anno.csv --result_file qwen_7b_64_lvb_results.json
python eval.py --anno_file vidmme_anno.csv --result_file qwen_7b_64_vidmme_results.json 
```


## Input Data Format

The model result JSON file must have the following structure:

```json
[
  {
    "id": "video_identifier",
    "frame_timestamps": [1.5, 3.2, 5.8, ]
  },
]
```


