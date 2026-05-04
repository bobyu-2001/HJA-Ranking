import json
import numpy as np
from scipy.stats import spearmanr


EPS = 1e-12



def comparisons_to_aggregated(comparisons, N, K):
    n_ijk = np.zeros((K, N, N), dtype=float)
    y_ijk = np.zeros((K, N, N), dtype=float)

    for k, i, j, y in comparisons:
        if i >= j:
            raise ValueError(f"comparison indices must satisfy i < j, got {(i, j)}")
        n_ijk[k, i, j] += 1.0
        y_ijk[k, i, j] += float(y)

    return n_ijk, y_ijk



def build_real_dataset(records, source_path=None):
    if not isinstance(records, list):
        raise ValueError("expected list of records")

    processed = []
    unknown_count = 0
    tie_count = 0
    label_counts = {"a": 0, "b": 0, "c": 0, "unknown": 0, "other": 0}

    judge_names = sorted({record["judge_model"] for record in records if "judge_model" in record})
    item_names = sorted(
        {
            model_name
            for record in records
            for model_name in (record.get("model_a"), record.get("model_b"))
            if model_name is not None
        }
    )
    judge_to_idx = {name: idx for idx, name in enumerate(judge_names)}
    item_to_idx = {name: idx for idx, name in enumerate(item_names)}

    for idx, record in enumerate(records):
        label = record.get("judge_preferred_model")
        normalized_label = str(label).lower() if label is not None else "unknown"
        if normalized_label not in {"a", "b", "c", "unknown"}:
            label_counts["other"] += 1
            continue
        label_counts[normalized_label] += 1
        if normalized_label == "unknown":
            unknown_count += 1
            continue
        if normalized_label == "c":
            tie_count += 1

        judge_name = record.get("judge_model")
        model_a = record.get("model_a")
        model_b = record.get("model_b")
        if judge_name not in judge_to_idx or model_a not in item_to_idx or model_b not in item_to_idx:
            continue
        if model_a == model_b:
            continue

        i_a = item_to_idx[model_a]
        i_b = item_to_idx[model_b]
        i, j = sorted((i_a, i_b))
        if normalized_label == "c":
            y = 0.5
        elif normalized_label == "a":
            y = int(i_a == i)
        else:
            y = int(i_b == i)

        processed.append(
            {
                "record_index": idx,
                "question_id": record.get("question_id"),
                "judge_model": judge_name,
                "model_a": model_a,
                "model_b": model_b,
                "judge_confidence": record.get("judge_confidence"),
                "preferred_label": normalized_label,
                "k": judge_to_idx[judge_name],
                "i": i,
                "j": j,
                "y": y,
            }
        )

    return {
        "records": records,
        "processed": processed,
        "judge_names": judge_names,
        "item_names": item_names,
        "judge_to_idx": judge_to_idx,
        "item_to_idx": item_to_idx,
        "summary": {
            "source_path": source_path,
            "total_records": len(records),
            "usable_records": len(processed),
            "unknown_count": unknown_count,
            "tie_count": tie_count,
            "label_counts": label_counts,
            "num_judges": len(judge_names),
            "num_items": len(item_names),
        },
    }



def load_real_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return build_real_dataset(records, source_path=json_path)



def build_real_dataset_from_records(records):
    return build_real_dataset(records)






def split_real_dataset(processed_records, test_ratio=0.2, random_seed=42):
    if not processed_records:
        raise ValueError("cannot split an empty processed dataset")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must lie in (0,1), got {test_ratio}")

    rng = np.random.default_rng(random_seed)
    indices = np.arange(len(processed_records))
    rng.shuffle(indices)

    test_size = max(1, int(round(len(processed_records) * test_ratio)))
    test_size = min(test_size, len(processed_records) - 1)
    test_indices = set(indices[:test_size].tolist())

    train_records = [processed_records[idx] for idx in range(len(processed_records)) if idx not in test_indices]
    test_records = [processed_records[idx] for idx in range(len(processed_records)) if idx in test_indices]
    return train_records, test_records



def processed_records_to_comparisons(processed_records):
    return [(entry["k"], entry["i"], entry["j"], entry["y"]) for entry in processed_records]



def processed_records_to_aggregated(processed_records, N, K):
    comparisons = processed_records_to_comparisons(processed_records)
    return comparisons_to_aggregated(comparisons, N, K)



def bootstrap_processed_records(processed_records, random_seed=42):
    if not processed_records:
        raise ValueError("cannot bootstrap an empty processed dataset")
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, len(processed_records), size=len(processed_records))
    return [processed_records[int(idx)] for idx in indices]



def summarize_rank_distribution(rankings, item_names, top_k):
    if not rankings:
        return {}
    top_k = max(1, min(int(top_k), len(item_names)))
    rank_values = {item_name: [] for item_name in item_names}
    top_k_hits = {item_name: 0 for item_name in item_names}

    for ranking_summary in rankings:
        item_to_rank = ranking_summary["item_to_rank"]
        for item_name in item_names:
            rank_value = int(item_to_rank[item_name])
            rank_values[item_name].append(rank_value)
            top_k_hits[item_name] += int(rank_value <= top_k)

    out = {}
    num_rankings = len(rankings)
    for item_name in item_names:
        values = np.asarray(rank_values[item_name], dtype=float)
        out[item_name] = {
            "mean_rank": float(np.mean(values)),
            "median_rank": float(np.median(values)),
            "rank_p05": float(np.quantile(values, 0.05)),
            "rank_p95": float(np.quantile(values, 0.95)),
            "top_k_frequency": float(top_k_hits[item_name] / num_rankings),
        }
    return out



def compute_top_k_stability(rankings, baseline_ranking, top_k, exact_match=True):
    if not baseline_ranking:
        raise ValueError("baseline_ranking must be non-empty")
    top_k = max(1, min(int(top_k), len(baseline_ranking)))
    baseline_top_k = baseline_ranking[:top_k]
    baseline_top_k_set = set(baseline_top_k)
    item_names = list(baseline_ranking)
    top_k_hits = {item_name: 0 for item_name in item_names}
    exact_matches = 0
    jaccard_scores = []

    for ranking_summary in rankings:
        current_top_k = ranking_summary["ranking"][:top_k]
        current_top_k_set = set(current_top_k)
        if exact_match:
            exact_matches += int(current_top_k == baseline_top_k)
        else:
            exact_matches += int(current_top_k_set == baseline_top_k_set)
        union_size = len(baseline_top_k_set | current_top_k_set)
        jaccard_scores.append(len(baseline_top_k_set & current_top_k_set) / union_size if union_size > 0 else 1.0)
        for item_name in current_top_k:
            top_k_hits[item_name] += 1

    num_rankings = len(rankings)
    return {
        "top_k": top_k,
        "baseline_top_k": baseline_top_k,
        "top_k_frequency": {
            item_name: float(top_k_hits[item_name] / num_rankings)
            for item_name in item_names
        },
        "all_bootstrap_match_baseline_top_k": bool(exact_matches == num_rankings),
        "exact_match_order_sensitive": bool(exact_match),
        "mean_jaccard_vs_baseline": float(np.mean(jaccard_scores)) if jaccard_scores else 1.0,
        "exact_match_rate_vs_baseline_top_k": float(exact_matches / num_rankings) if num_rankings > 0 else 1.0,
    }





def make_noisy_judge_records(source_records, judge_name, random_seed=42):
    rng = np.random.default_rng(random_seed)
    noisy_records = []
    for record in source_records:
        noisy_records.append(
            {
                "question_id": record["question_id"],
                "model_a": record["model_a"],
                "model_b": record["model_b"],
                "judge_model": judge_name,
                "judge_preferred_model": "a" if int(rng.binomial(1, 0.5)) == 1 else "b",
                "judge_confidence": 0.5,
            }
        )
    return noisy_records



def rank_items_from_mu(mu, item_names):
    mu = np.asarray(mu, dtype=float)
    order = np.argsort(-mu)
    ranking = [item_names[idx] for idx in order]
    item_to_rank = {item_names[idx]: rank + 1 for rank, idx in enumerate(order)}
    return {
        "mu": {item_names[idx]: float(mu[idx]) for idx in range(len(item_names))},
        "ranking": ranking,
        "item_to_rank": item_to_rank,
    }



def pairwise_rank_agreement(reference_ranking, current_ranking):
    ref_rank = {item: idx for idx, item in enumerate(reference_ranking)}
    cur_rank = {item: idx for idx, item in enumerate(current_ranking)}
    items = reference_ranking
    total = 0
    correct = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            total += 1
            ref_sign = np.sign(ref_rank[items[i]] - ref_rank[items[j]])
            cur_sign = np.sign(cur_rank[items[i]] - cur_rank[items[j]])
            correct += int(ref_sign == cur_sign)
    return float(correct / total) if total > 0 else 1.0



def compute_rank_shift(reference_summary, current_summary):
    reference_ranking = reference_summary["ranking"]
    current_ranking = current_summary["ranking"]
    reference_mu = np.array([reference_summary["mu"][item] for item in reference_ranking], dtype=float)
    current_mu = np.array([current_summary["mu"][item] for item in reference_ranking], dtype=float)
    spearman, _ = spearmanr(reference_mu, current_mu)
    rank_shift = {
        item: int(current_summary["item_to_rank"][item] - reference_summary["item_to_rank"][item])
        for item in reference_ranking
    }
    return {
        "spearman": float(spearman) if len(reference_ranking) > 1 else 1.0,
        "pairwise_agreement": pairwise_rank_agreement(reference_ranking, current_ranking),
        "top_1_changed": bool(reference_ranking[0] != current_ranking[0]) if reference_ranking else False,
        "rank_shift": rank_shift,
    }
