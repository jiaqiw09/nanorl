from __future__ import annotations

import signal
import time
import traceback

from .actor import MegatronActor
from .checkpoint import load_checkpoint, model_metadata, restore_rng, save_checkpoint, seed_training
from .config import parse_args
from .grpo import group_advantages
from .reward import load_reward, load_examples
from .rollout import SGLangRollout, prompt_token_ids, validate_batch
from .run_log import RunLog
from .sampler import ResumableSampler
from .sync import SGLangWeightSync


def main() -> None:
    from transformers import AutoTokenizer

    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = config.output_dir.resolve().parent
    log = RunLog(run_dir, config)
    # A fresh session avoids colliding with a previously published policy version.
    sync_dir = config.output_dir / log.session_id
    checkpoint_dir = run_dir / "checkpoints"
    actor, weight_sync = None, None
    completed_step, stage = 0, "initialize"
    training_checkpoint = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}; resume from the last committed checkpoint")

    old_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        log.write("start", config_path=str(log.config_path), resume=str(config.resume) if config.resume else None)
        sampler = ResumableSampler(load_examples(config.data_path), seed=config.seed, shuffle=config.shuffle)
        metadata = model_metadata(config.model_path)
        reward = load_reward(config.reward)
        tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=False)
        actor = MegatronActor(config.model_path, config.learning_rate, backend=config.actor_backend, dtype=config.actor_dtype)
        seed_training(config.seed)
        saved_rng = None
        if config.resume is not None:
            stage = "restore"
            completed_step, saved_rng, training_checkpoint = load_checkpoint(
                config.resume, actor=actor, sampler=sampler, config=config, metadata=metadata,
            )
            log.write("resume", step=completed_step, policy_version=actor.policy_version,
                      training_checkpoint=str(training_checkpoint), data_fingerprint=sampler.fingerprint)

        rollout = SGLangRollout(config.sglang_url, config.max_new_tokens, config.max_sequence_length, seed=config.seed)
        stage = "initial_sync"
        if config.weight_sync == "nccl":
            from .sync_nccl import NCCLWeightSync
            weight_sync = NCCLWeightSync(config.sglang_url, sync_dir, dtype=config.rollout_dtype,
                                         bucket_mb=config.sync_bucket_mb, timeout=config.sync_timeout)
        else:
            weight_sync = SGLangWeightSync(config.sglang_url, sync_dir)
            weight_sync.timeout = config.sync_timeout
        inference_weights = weight_sync.publish(actor, version=actor.policy_version)
        # Initialization/export/communication setup must not consume the restored RNG stream.
        if saved_rng is not None:
            restore_rng(saved_rng)
        log.write("initial_sync", step=completed_step, policy_version=actor.policy_version,
                  inference_weights=str(inference_weights) if inference_weights else None, **weight_sync.last_metrics)
        if config.resume is None:
            stage = "checkpoint"
            training_checkpoint = save_checkpoint(checkpoint_dir, session_id=log.session_id, step=0,
                                                   actor=actor, sampler=sampler, config=config, metadata=metadata)
            log.write("checkpoint", step=0, training_checkpoint=str(training_checkpoint))

        for step in range(completed_step + 1, config.steps + 1):
            step_start = time.monotonic()
            stage = "sample"
            prompt_groups, examples_by_group = [], {}
            selected = sampler.take(config.prompts_per_step)
            for prompt_index, (_, example) in enumerate(selected):
                ids = prompt_token_ids(tokenizer, example.prompt)
                group_id = f"step-{step}/prompt-{prompt_index}"
                prompt_groups.append((group_id, ids))
                examples_by_group[group_id] = example
            started = time.monotonic()
            samples = rollout.generate_batch(prompt_groups, group_size=config.group_size,
                                             policy_version=weight_sync.committed_version,
                                             concurrency=config.rollout_concurrency)
            rollout_seconds = time.monotonic() - started
            stage, started = "reward", time.monotonic()
            rewards = [float(reward(examples_by_group[sample.group_id], sample.response_text)) for sample in samples]
            validate_batch(samples, config.group_size, actor.policy_version)
            advantages = group_advantages(rewards, config.group_size, group_ids=[sample.group_id for sample in samples])
            reward_seconds = time.monotonic() - started

            stage, started = "update", time.monotonic()
            update = actor.update(samples, advantages, config.clip_ratio, config.group_size,
                                  microbatch_size=config.train_microbatch_size)
            update_seconds = time.monotonic() - started
            stage = "sync"
            sync_metrics = {"sync_backend": config.weight_sync, "sync_seconds": 0.0, "sync_skipped": True}
            if update["update_applied"]:
                inference_weights = weight_sync.publish(actor, version=actor.policy_version)
                sync_metrics = {**weight_sync.last_metrics, "sync_skipped": False}

            # Sampler, actor and published policy now describe one completed round.
            completed_step = step
            checkpoint_seconds, checkpoint_saved = 0.0, False
            if step % config.checkpoint_every == 0 or step == config.steps:
                stage, started = "checkpoint", time.monotonic()
                training_checkpoint = save_checkpoint(checkpoint_dir, session_id=log.session_id, step=step,
                                                       actor=actor, sampler=sampler, config=config, metadata=metadata)
                checkpoint_seconds, checkpoint_saved = time.monotonic() - started, True
            stage = "log"
            lengths = [len(s.response_ids) for s in samples]
            mean_reward = sum(rewards) / len(rewards)
            log.write("step", step=step, mean_reward=mean_reward, reward_min=min(rewards), reward_max=max(rewards),
                      reward_std=(sum((r - mean_reward)**2 for r in rewards) / len(rewards))**0.5,
                      **update, **sync_metrics, rollout_seconds=rollout_seconds, reward_seconds=reward_seconds,
                      update_seconds=update_seconds, checkpoint_seconds=checkpoint_seconds,
                      step_seconds=time.monotonic() - step_start,
                      nonzero_advantages=sum(x != 0 for x in advantages),
                      response_tokens=sum(lengths), response_length_mean=sum(lengths) / len(lengths),
                      response_length_min=min(lengths), response_length_max=max(lengths),
                      truncated_responses=sum(s.truncated for s in samples),
                      active_group_fraction=len({s.group_id for s, a in zip(samples, advantages) if a != 0}) / config.prompts_per_step,
                      dataset_indices=[index for index, _ in selected], sampler_epoch=sampler.epoch, sampler_cursor=sampler.cursor,
                      checkpoint_saved=checkpoint_saved, training_checkpoint=str(training_checkpoint),
                      inference_weights=str(inference_weights) if inference_weights else None)
        log.write("complete", step=completed_step, policy_version=actor.policy_version,
                  training_checkpoint=str(training_checkpoint))
    except BaseException as exc:
        log.write("error", stage=stage, completed_step=completed_step,
                  training_checkpoint=str(training_checkpoint) if training_checkpoint else None,
                  error_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            try:
                if weight_sync is not None:
                    weight_sync.close()
            finally:
                if actor is not None:
                    actor.close()
        finally:
            log.close()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    main()
