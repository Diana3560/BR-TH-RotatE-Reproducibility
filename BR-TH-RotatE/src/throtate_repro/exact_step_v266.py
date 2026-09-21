from __future__ import annotations

"""A generic exact-step sLCWA loop for ordinary and reciprocal training."""

import itertools

try:
    from pykeen.training import SLCWATrainingLoop
except ImportError as exc:  # Keep the rest of the package importable without PyKEEN.
    SLCWATrainingLoop = None
    _PYKEEN_IMPORT_ERROR: ImportError | None = exc
else:
    _PYKEEN_IMPORT_ERROR = None


class _LimitedBatches:
    def __init__(self, source, limit: int) -> None:
        self.source = source
        self.limit = int(limit)

    def __iter__(self):
        yield from itertools.islice(iter(self.source), self.limit)

    def __len__(self) -> int:
        return min(len(self.source), self.limit)


if SLCWATrainingLoop is not None:

    class ExactStepSLCWATrainingLoop(SLCWATrainingLoop):
        """Truncate only the final epoch and audit the exact update count."""

        def __init__(
            self,
            *,
            exact_max_steps: int,
            steps_per_full_epoch: int,
            final_epoch_batches: int,
            exact_num_epochs: int,
            **kwargs,
        ) -> None:
            super().__init__(**kwargs)
            self.exact_max_steps = int(exact_max_steps)
            self.steps_per_full_epoch = int(steps_per_full_epoch)
            self.final_epoch_batches = int(final_epoch_batches)
            self.exact_num_epochs = int(exact_num_epochs)
            self.v266_optimizer_steps = 0

        def _train_epoch(self, *, batches, epoch: int, **kwargs):
            if int(epoch) == self.exact_num_epochs:
                batches = _LimitedBatches(batches, self.final_epoch_batches)
            n_batches = len(batches)
            loss = super()._train_epoch(batches=batches, epoch=epoch, **kwargs)
            if not bool(kwargs.get("only_size_probing", False)):
                self.v266_optimizer_steps += int(n_batches)
            return loss

else:
    ExactStepSLCWATrainingLoop = None


def build_exact_step_training_loop_class():
    if ExactStepSLCWATrainingLoop is None:
        raise RuntimeError("PyKEEN is not installed in this environment") from _PYKEEN_IMPORT_ERROR
    return ExactStepSLCWATrainingLoop
