"""Run the pinned upstream Task 1 trainer and retain its fitted models."""
from pathlib import Path
import sys

REPO = Path('/root/autodl-tmp/C_Problem/team-code')
sys.path.insert(0, str(REPO))
from scripts.task1_v4 import train_task1_v4 as trainer

original_fold = trainer.run_fold
original_fit = trainer.fit_classifier
context = {}


def run_fold(args, fold, directory, run_id):
    context['directory'] = directory
    print(f"FOLD_START {fold['fold']}", flush=True)
    return original_fold(args, fold, directory, run_id)


def fit_classifier(*args, **kwargs):
    model = original_fit(*args, **kwargs)
    # Saving the already fitted Booster changes neither fitting nor prediction.
    model.booster_.save_model(str(context['directory'] / 'model.txt'),
                             num_iteration=model.best_iteration_)
    return model


trainer.run_fold = run_fold
trainer.fit_classifier = fit_classifier
if __name__ == '__main__':
    trainer.main()
