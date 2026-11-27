"""Training-only estimation and independent trial-wise Viterbi decoding."""

import numpy as np


def _sequences(frame):
    """Yield ordered row positions, resetting at each trial or missing window."""
    keys = ["subject", "trial", "source_start"]
    coordinates = frame[keys].reset_index(drop=True)
    if coordinates.isna().any().any() or coordinates.duplicated().any():
        raise ValueError("Trial coordinates must be nonmissing and unique")
    if "stride_frames" in frame:
        coordinates["stride_frames"] = frame["stride_frames"].to_numpy()
    coordinates = coordinates.sort_values(keys, kind="stable")
    for _, group in coordinates.groupby(keys[:2], sort=False):
        positions = group.index.to_numpy()
        if "stride_frames" not in group:
            yield positions
            continue
        gaps = np.diff(group["source_start"].to_numpy())
        expected = group["stride_frames"].to_numpy()[:-1]
        if np.any(expected <= 0):
            raise ValueError("Window stride must be positive")
        for segment in np.split(positions, np.flatnonzero(gaps != expected) + 1):
            yield segment


def fit_sticky_hmm(train_df, dt_seconds=1.0, n_classes=13, smoothing=1.0):
    """Fit durations, transitions and prior from the provided training windows."""
    if "stride_frames" in train_df:
        strides = train_df["stride_frames"].to_numpy(dtype=float)
        if (not len(strides) or not np.isfinite(strides).all() or strides[0] <= 0
                or not strides[0].is_integer() or not (strides == strides[0]).all()):
            raise ValueError("Training windows must share one positive integer stride_frames")
        dt_seconds = float(strides[0] / 60.0)
    if not np.isfinite(dt_seconds) or dt_seconds <= 0 or not np.isfinite(smoothing) or (
        smoothing <= 0 or n_classes < 2
    ):
        raise ValueError("Require a positive time step, positive smoothing and at least two classes")
    label_column = "majority_label" if "majority_label" in train_df else "y_true"
    source_labels = train_df[label_column].to_numpy()
    if not len(source_labels) or not np.isin(source_labels, np.arange(n_classes)).all():
        raise ValueError("Training labels must be nonempty and in range")
    labels = source_labels.astype(np.int64)
    transitions = np.full((n_classes, n_classes), float(smoothing))
    np.fill_diagonal(transitions, 0)
    run_total = np.zeros(n_classes)
    run_count = np.zeros(n_classes)
    for positions in _sequences(train_df):
        trial = labels[positions]
        changes = np.flatnonzero(trial[1:] != trial[:-1]) + 1
        starts = np.r_[0, changes]
        classes = trial[starts]
        np.add.at(run_total, classes, np.diff(np.r_[starts, len(trial)]))
        np.add.at(run_count, classes, 1)
        np.add.at(transitions, (classes[:-1], classes[1:]), 1)
    backstop = run_total.sum() / run_count.sum()
    duration = dt_seconds * np.divide(run_total, run_count, out=np.full(n_classes, backstop),
                                      where=run_count > 0)
    stay = np.exp(-dt_seconds / duration)
    transitions /= transitions.sum(axis=1, keepdims=True)
    transitions *= (1 - stay)[:, None]
    np.fill_diagonal(transitions, stay)
    prior = np.bincount(labels, minlength=n_classes) + smoothing
    return {"tau_seconds": duration, "transition": transitions,
            "prior": prior / prior.sum(), "dt_seconds": dt_seconds}


def viterbi_decode_trials(df, prob_cols, hmm_params):
    """Decode independent trials and return predicted labels in original row order."""
    probabilities = df[list(prob_cols)].to_numpy(dtype=float)
    prior = np.asarray(hmm_params["prior"])
    transition = np.asarray(hmm_params["transition"])
    n_classes = len(prior)
    if probabilities.shape[1] != n_classes or transition.shape != (n_classes, n_classes):
        raise ValueError("Probabilities and HMM parameters have different class counts")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (
        probabilities.sum(axis=1) <= 0
    ).any():
        raise ValueError("Each window needs finite, nonnegative class probabilities")
    log_prob = np.log(np.maximum(probabilities, 1e-15))
    log_prior = np.log(prior)
    log_transition = np.log(transition)
    final = np.empty(len(df), dtype=np.int64)
    classes = np.arange(n_classes)
    for positions in _sequences(df):
        emissions = log_prob[positions]
        predecessor = np.empty((len(positions), n_classes), dtype=np.int64)
        score = log_prior + emissions[0]
        for step in range(1, len(positions)):
            candidates = score[:, None] + log_transition
            predecessor[step] = candidates.argmax(axis=0)
            score = candidates[predecessor[step], classes] + emissions[step]
        label = int(score.argmax())
        for step in range(len(positions) - 1, -1, -1):
            final[positions[step]] = label
            if step:
                label = predecessor[step, label]
    return final
