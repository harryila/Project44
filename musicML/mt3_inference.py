"""Local MT3 inference wrapper.

Adapted from the MT3 Colab notebook:
https://github.com/magenta/mt3/blob/main/mt3/colab/music_transcription_with_transformers.ipynb
"""

import functools
import os
import sys
from pathlib import Path

import jax
# JAX 0.9 removed jax.tree_* in favor of jax.tree.* — shim for old t5x code
if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree.map
    jax.tree_leaves = jax.tree.leaves
    jax.tree_structure = jax.tree.structure
    jax.tree_unflatten = jax.tree.unflatten
    jax.tree_flatten = jax.tree.flatten
    jax.tree_transpose = jax.tree.transpose
    jax.tree_all = jax.tree.all

import gin
import librosa
import note_seq
import numpy as np
import seqio
import t5
import t5x
import tensorflow.compat.v2 as tf

from mt3 import metrics_utils
from mt3 import models
from mt3 import network
from mt3 import note_sequences
from mt3 import preprocessors
from mt3 import spectrograms
from mt3 import vocabularies

import nest_asyncio
nest_asyncio.apply()

SAMPLE_RATE = 16000

SCRIPT_DIR = Path(__file__).resolve().parent
MT3_DIR = SCRIPT_DIR / "mt3"
GIN_DIR = MT3_DIR / "mt3" / "gin"
CHECKPOINT_DIR = MT3_DIR / "checkpoints"


class InferenceModel:
    """Wrapper of T5X model for music transcription."""

    def __init__(self, checkpoint_path, model_type="ismir2021"):
        if model_type == "ismir2021":
            num_velocity_bins = 127
            self.encoding_spec = note_sequences.NoteEncodingSpec
            self.inputs_length = 512
        elif model_type == "mt3":
            num_velocity_bins = 1
            self.encoding_spec = note_sequences.NoteEncodingWithTiesSpec
            self.inputs_length = 256
        else:
            raise ValueError(f"unknown model_type: {model_type}")

        gin_files = [
            str(GIN_DIR / "model.gin"),
            str(GIN_DIR / f"{model_type}.gin"),
        ]

        self.batch_size = 8
        self.outputs_length = 1024
        self.sequence_length = {
            "inputs": self.inputs_length,
            "targets": self.outputs_length,
        }

        self.partitioner = t5x.partitioning.PjitPartitioner(num_partitions=1)

        self.spectrogram_config = spectrograms.SpectrogramConfig()
        self.codec = vocabularies.build_codec(
            vocab_config=vocabularies.VocabularyConfig(
                num_velocity_bins=num_velocity_bins
            )
        )
        self.vocabulary = vocabularies.vocabulary_from_codec(self.codec)
        self.output_features = {
            "inputs": seqio.ContinuousFeature(dtype=tf.float32, rank=2),
            "targets": seqio.Feature(vocabulary=self.vocabulary),
        }

        self._parse_gin(gin_files)
        self.model = self._load_model()
        self.restore_from_checkpoint(checkpoint_path)

    @property
    def input_shapes(self):
        return {
            "encoder_input_tokens": (self.batch_size, self.inputs_length),
            "decoder_input_tokens": (self.batch_size, self.outputs_length),
        }

    def _parse_gin(self, gin_files):
        gin_bindings = [
            "from __gin__ import dynamic_registration",
            "from mt3 import vocabularies",
            "VOCAB_CONFIG=@vocabularies.VocabularyConfig()",
            "vocabularies.VocabularyConfig.num_velocity_bins=%NUM_VELOCITY_BINS",
        ]
        with gin.unlock_config():
            gin.parse_config_files_and_bindings(
                gin_files, gin_bindings, finalize_config=False
            )

    def _load_model(self):
        model_config = gin.get_configurable(network.T5Config)()
        module = network.Transformer(config=model_config)
        return models.ContinuousInputsEncoderDecoderModel(
            module=module,
            input_vocabulary=self.output_features["inputs"].vocabulary,
            output_vocabulary=self.output_features["targets"].vocabulary,
            optimizer_def=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0),
            input_depth=spectrograms.input_depth(self.spectrogram_config),
        )

    def restore_from_checkpoint(self, checkpoint_path):
        train_state_initializer = t5x.utils.TrainStateInitializer(
            optimizer_def=self.model.optimizer_def,
            init_fn=self.model.get_initial_variables,
            input_shapes=self.input_shapes,
            partitioner=self.partitioner,
        )

        restore_checkpoint_cfg = t5x.utils.RestoreCheckpointConfig(
            path=checkpoint_path, mode="specific", dtype="float32"
        )

        train_state_axes = train_state_initializer.train_state_axes
        self._predict_fn = self._get_predict_fn(train_state_axes)
        self._train_state = train_state_initializer.from_checkpoint_or_scratch(
            [restore_checkpoint_cfg], init_rng=jax.random.PRNGKey(0)
        )

    @functools.lru_cache()
    def _get_predict_fn(self, train_state_axes):
        def partial_predict_fn(params, batch, decode_rng):
            return self.model.predict_batch_with_aux(
                params, batch, decoder_params={"decode_rng": None}
            )

        return self.partitioner.partition(
            partial_predict_fn,
            in_axis_resources=(
                train_state_axes.params,
                t5x.partitioning.PartitionSpec("data"),
                None,
            ),
            out_axis_resources=t5x.partitioning.PartitionSpec("data"),
        )

    def predict_tokens(self, batch, seed=0):
        prediction, _ = self._predict_fn(
            self._train_state.params, batch, jax.random.PRNGKey(seed)
        )
        return self.vocabulary.decode_tf(prediction).numpy()

    def __call__(self, audio):
        """Infer note sequence from audio samples (16kHz numpy array)."""
        ds = self.audio_to_dataset(audio)
        ds = self.preprocess(ds)

        model_ds = self.model.FEATURE_CONVERTER_CLS(pack=False)(
            ds, task_feature_lengths=self.sequence_length
        )
        model_ds = model_ds.batch(self.batch_size)

        inferences = (
            tokens
            for batch in model_ds.as_numpy_iterator()
            for tokens in self.predict_tokens(batch)
        )

        predictions = []
        for example, tokens in zip(ds.as_numpy_iterator(), inferences):
            predictions.append(self.postprocess(tokens, example))

        result = metrics_utils.event_predictions_to_ns(
            predictions, codec=self.codec, encoding_spec=self.encoding_spec
        )
        return result["est_ns"]

    def audio_to_dataset(self, audio):
        frames, frame_times = self._audio_to_frames(audio)
        return tf.data.Dataset.from_tensors(
            {"inputs": frames, "input_times": frame_times}
        )

    def _audio_to_frames(self, audio):
        frame_size = self.spectrogram_config.hop_width
        padding = [0, frame_size - len(audio) % frame_size]
        audio = np.pad(audio, padding, mode="constant")
        frames = spectrograms.split_audio(audio, self.spectrogram_config)
        num_frames = len(audio) // frame_size
        times = np.arange(num_frames) / self.spectrogram_config.frames_per_second
        return frames, times

    def preprocess(self, ds):
        pp_chain = [
            functools.partial(
                t5.data.preprocessors.split_tokens_to_inputs_length,
                sequence_length=self.sequence_length,
                output_features=self.output_features,
                feature_key="inputs",
                additional_feature_keys=["input_times"],
            ),
            preprocessors.add_dummy_targets,
            functools.partial(
                preprocessors.compute_spectrograms,
                spectrogram_config=self.spectrogram_config,
            ),
        ]
        for pp in pp_chain:
            ds = pp(ds)
        return ds

    def postprocess(self, tokens, example):
        tokens = self._trim_eos(tokens)
        start_time = example["input_times"][0]
        start_time -= start_time % (1 / self.codec.steps_per_second)
        return {
            "est_tokens": tokens,
            "start_time": start_time,
            "raw_inputs": [],
        }

    @staticmethod
    def _trim_eos(tokens):
        tokens = np.array(tokens, np.int32)
        if vocabularies.DECODED_EOS_ID in tokens:
            tokens = tokens[: np.argmax(tokens == vocabularies.DECODED_EOS_ID)]
        return tokens


_cached_model = None


def _get_model(model_type="ismir2021"):
    """Return a cached InferenceModel, loading only on first call or type change."""
    global _cached_model
    if _cached_model is None or _cached_model[0] != model_type:
        checkpoint_path = str(CHECKPOINT_DIR / model_type)
        _cached_model = (model_type, InferenceModel(checkpoint_path, model_type))
    return _cached_model[1]


def transcribe_audio(audio_path: str, midi_path: str, model_type: str = "ismir2021"):
    """Transcribe an audio file to MIDI using MT3.

    Args:
        audio_path: Path to input audio file.
        midi_path: Path to output MIDI file.
        model_type: 'ismir2021' for piano or 'mt3' for multi-instrument.

    Returns:
        Number of notes detected.
    """
    import time

    t0 = time.time()
    print(f"  [MT3] Loading audio from {audio_path}...")
    sys.stdout.flush()
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    duration_sec = len(audio) / SAMPLE_RATE
    print(f"  [MT3] Audio loaded: {duration_sec:.1f}s, {len(audio)} samples ({time.time()-t0:.1f}s)")
    sys.stdout.flush()

    t1 = time.time()
    print(f"  [MT3] Loading model (checkpoint: {model_type})...")
    sys.stdout.flush()
    model = _get_model(model_type)
    print(f"  [MT3] Model loaded ({time.time()-t1:.1f}s)")
    sys.stdout.flush()

    t2 = time.time()
    print(f"  [MT3] Building spectrogram and splitting into segments...")
    sys.stdout.flush()
    ds = model.audio_to_dataset(audio)
    ds = model.preprocess(ds)
    ds = ds.cache()
    n_segments = sum(1 for _ in ds.as_numpy_iterator())
    print(f"  [MT3] {n_segments} segments to process ({time.time()-t2:.1f}s)")
    sys.stdout.flush()

    t3 = time.time()
    print(f"  [MT3] Running inference...")
    sys.stdout.flush()

    model_ds = model.model.FEATURE_CONVERTER_CLS(pack=False)(
        ds, task_feature_lengths=model.sequence_length
    )
    model_ds = model_ds.batch(model.batch_size)

    predictions = []
    ds_iter = iter(ds.as_numpy_iterator())
    batch_num = 0
    for batch in model_ds.as_numpy_iterator():
        batch_num += 1
        tb = time.time()
        tokens_batch = model.predict_tokens(batch)
        print(f"  [MT3]   Batch {batch_num}: {len(tokens_batch)} segments ({time.time()-tb:.1f}s)")
        sys.stdout.flush()
        for tokens in tokens_batch:
            try:
                example = next(ds_iter)
            except StopIteration:
                break
            predictions.append(model.postprocess(tokens, example))

    print(f"  [MT3] Inference complete ({time.time()-t3:.1f}s)")
    sys.stdout.flush()

    t4 = time.time()
    print(f"  [MT3] Decoding tokens to NoteSequence...")
    sys.stdout.flush()
    result = metrics_utils.event_predictions_to_ns(
        predictions, codec=model.codec, encoding_spec=model.encoding_spec
    )
    est_ns = result["est_ns"]
    print(f"  [MT3] {len(est_ns.notes)} notes decoded ({time.time()-t4:.1f}s)")
    sys.stdout.flush()

    note_seq.sequence_proto_to_midi_file(est_ns, midi_path)
    print(f"  [MT3] MIDI saved to {midi_path}")
    print(f"  [MT3] Total time: {time.time()-t0:.1f}s")
    sys.stdout.flush()

    return len(est_ns.notes)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <audio_file> [output.mid]")
        sys.exit(1)

    audio_file = sys.argv[1]
    midi_file = sys.argv[2] if len(sys.argv) > 2 else "mt3_output.mid"

    print(f"Transcribing {audio_file} -> {midi_file}")
    sys.stdout.flush()
    n = transcribe_audio(audio_file, midi_file)
    print(f"Done. {n} notes detected.")
