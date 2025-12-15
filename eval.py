import json
import csv
import argparse
import numpy as np


def timestamp_to_seconds(timestamp):
    if len(timestamp.split(":")) == 3:
        h, m, s = timestamp.split(":")
        total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    elif len(timestamp.split(":")) == 2:
        m, s = timestamp.split(":")
        total_seconds = int(m) * 60 + float(s)
    else:
        raise Exception("Wrong timestamp format")
    return total_seconds


def main(args):
    with open(args.result_file) as f:
        results = json.load(f)

    gt_scenes = {}
    with open(args.anno_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = timestamp_to_seconds(row["start"])
            end = timestamp_to_seconds(row["end"]) + 1
            if row["id"] not in gt_scenes:
                gt_scenes[row["id"]] = {}
            if row["scene_id"] not in gt_scenes[row["id"]]:
                gt_scenes[row["id"]][row["scene_id"]] = []
            gt_scenes[row["id"]][row["scene_id"]].append([start, end])

    kfs_metrics = []
    for res in results:
        if res["id"] not in gt_scenes:
            continue

        timestamps = np.array(res["frame_timestamps"])
        scenes = [np.array(scene_i) for scene_i in gt_scenes[res["id"]].values()]

        scenes_flatten = np.concatenate(scenes, 0)
        n_tp = np.any((timestamps[:, None] >= scenes_flatten[:, 0]) * (timestamps[:, None] <= scenes_flatten[:, 1]), axis=1).sum()
        kfr = n_tp / len(timestamps) # Key Frame Rate

        n_hit = np.sum([
            np.any((timestamps[:, None] >= scene_i[:, 0]) * (timestamps[:, None] <= scene_i[:, 1]))
            for scene_i in scenes
        ])
        shr = n_hit / len(scenes) # Scene Hit Rate

        n_tp_per_scene = np.array([
            np.any((timestamps[:, None] >= scene_i[:, 0]) * (timestamps[:, None] <= scene_i[:, 1]), axis=1).sum()
            for scene_i in scenes
        ])
        scene_lens = np.array([
            np.sum(scene_i[:, 1] - scene_i[:, 0])
            for scene_i in scenes
        ])
        thres_per_scene = np.clip(
            np.floor(
                np.minimum(
                    scene_lens / np.sum(scene_lens) * n_tp,
                    np.ones(len(scenes)) / len(scenes) * n_tp
                )
            ),
            a_min=1, a_max=None
        )
        bsr = np.mean(n_tp_per_scene >= thres_per_scene) # Balanced Scene Recall

        if n_tp > 0:
            tp_dist = n_tp_per_scene / np.sum(n_tp_per_scene)
            len_dists = scene_lens[:, None] ** (0.1 * np.arange(11))
            sims = tp_dist @ len_dists / (np.linalg.norm(tp_dist) * np.linalg.norm(len_dists, axis=0))
            bds = np.max(sims) # Balanced Distribution Similarity
        else:
            bds = 0

        kfs_metrics.append(
            {
                "id": res["id"],
                "kfr": kfr,
                "shr": shr,
                "bsr": bsr,
                "bds": bds,
                "ukss": (kfr * bsr * bds) ** (1 / 3),
            }
        )

    mean_kfr = np.mean(np.array([m["kfr"] for m in kfs_metrics]))
    mean_shr = np.mean(np.array([m["shr"] for m in kfs_metrics]))
    ukss = np.clip(np.array([m["ukss"] for m in kfs_metrics]), a_min=1e-2, a_max=None)
    mean_ukss = np.exp(np.mean(np.log(ukss)))

    print("UKSS = {:.4f}".format(mean_ukss))
    print("Key Frame Rate = {:.4f}".format(mean_kfr))
    print("Scene Hit Rate = {:.4f}".format(mean_shr))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_file", type=str, default="./vidmme_anno.csv")
    parser.add_argument("--result_file", type=str, default="./qwen_7b_64_vidmme_results.json")
    args = parser.parse_args()
    main(args)