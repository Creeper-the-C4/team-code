# Held-out sensor selection (ranking_prefix)

Gain ranking on one full-layout training fit determines the ranking_prefix comparison. Shared feature gains are divided equally among their dependent sensors. Each smaller prefix gets a separate training fit with validation early stopping. Canonical sensor order breaks remaining ties.

The outer test participants are predicted once per selected layout size. The reported fold standard deviations use ddof=1. Sensor frequencies count folds. If classes share the minimum F1, the lowest activity ID is reported. Ranks count removal steps for backward selection (the final survivor ranks last), or entry steps for forward selection and gain prefixes. Step times include fitting and prediction. The Pareto flag compares participant-equal held-out scores against smaller layouts and is descriptive. Use the corresponding fold of per_fold_selected_subsets.json in downstream held-out work.
