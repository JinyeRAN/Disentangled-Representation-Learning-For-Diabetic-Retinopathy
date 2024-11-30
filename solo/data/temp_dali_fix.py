import logging
import math

import numpy as np
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


class TempDALIGenericIterator(DALIGenericIterator):
    """Temporary fix to avoid epoch-skiping when setting last_batch_policy=Drop."""

    def _advance_and_check_drop_last(self, dry_run=False, end_iteration=True):
        counter = self._counter
        should_end = False
        if self._reader_name:
            counter += self.batch_size
            if self._last_batch_policy == LastBatchPolicy.DROP:
                should_end = np.any(self._counter_per_gpu + counter > self._shard_sizes_per_gpu)
        else:
            counter += self._num_gpus * self.batch_size
            if self._last_batch_policy == LastBatchPolicy.DROP:
                should_end = counter > self._size

        if not dry_run:
            self._counter = counter
            if should_end and end_iteration:
                self._end_iteration()

        return should_end

    def reset(self):
        if self._last_batch_policy == LastBatchPolicy.DROP and not ():
            should_end = self._advance_and_check_drop_last(dry_run=True, end_iteration=False)
            already_ended = self._size > 0 and self._counter >= self._size
            if should_end and not already_ended:
                self._get_outputs()
                self._schedule_runs()
                self._advance_and_check_drop_last(end_iteration=False)

        if self._counter >= self._size or self._size < 0:
            if self._last_batch_policy == LastBatchPolicy.FILL and not self._last_batch_padded:
                if self._reader_name:
                    self._counter -= min(self._counter_per_gpu)
                    self._counter_per_gpu = self._counter_per_gpu + self._counter
                    self._counter_per_gpu = self._counter_per_gpu - self._shard_sizes_per_gpu
                    self._counter = min(self._counter_per_gpu)
                else:
                    # legacy way
                    self._counter = self._counter % self._size
            else:
                self._counter = 0
            # advance to the next shard
            if self._reader_name:
                if not self._is_stick_to_shard:
                    # move shards id for wrapped pipelines
                    self._shards_id = (self._shards_id + 1) % self._shards_num
                # revaluate _size
                if self._last_batch_policy == LastBatchPolicy.FILL and not self._last_batch_padded:
                    # move all shards ids GPU ahead
                    if not self._is_stick_to_shard:
                        self._shard_sizes_per_gpu = np.roll(self._shard_sizes_per_gpu, 1)
                    read_in_next_epoch = self._shard_sizes_per_gpu - self._counter_per_gpu
                    self._size = (math.ceil(max(read_in_next_epoch) / self.batch_size) * self.batch_size)

                    if self._size == 0:

                        self._counter_per_gpu = np.zeros(self._shards_num, dtype=np.int64)

                        self._counter = 0
                        self._shard_sizes_per_gpu = np.roll(self._shard_sizes_per_gpu, 1)
                        self._size = (
                            math.ceil(max(self._shard_sizes_per_gpu) / self.batch_size)
                            * self.batch_size
                        )

            for p in self._pipes:
                p.reset()
                if p.empty():
                    with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                        p.schedule_run()
        else:
            logging.warning(
                "DALI iterator does not support resetting while epoch is not finished. \
                             Ignoring..."
            )
