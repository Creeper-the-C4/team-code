# Held-out sensor selection (forward_greedy)

The exact wrapper evaluates every candidate addition or removal on validation participants. Participant-equal Macro-F1 determines the winner and pooled Macro-F1 breaks ties. Canonical sensor order breaks remaining ties.

The outer test participants are predicted once per selected layout size. The reported fold standard deviations use ddof=1. Sensor frequencies count folds. If classes share the minimum F1, the lowest activity ID is reported. Ranks count removal steps for backward selection (the final survivor ranks last), or entry steps for forward selection and gain prefixes. Step times include fitting and prediction. The Pareto flag compares participant-equal held-out scores against smaller layouts and is descriptive. Use the corresponding fold of per_fold_selected_subsets.json in downstream held-out work.
