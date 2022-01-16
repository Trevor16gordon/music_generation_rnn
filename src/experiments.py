"""
Experiments in this module group together 
    - Loading different types of music
    - Different variations of models for training
    - Different variations of prediction functions for music prediction


In this model Experiments 1 - 9 are experiments that build up to the final models 10 and 11.
10 and 11 differ in which music is loaded.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from src.utils.midi_support import (
    MidiSupport,
    RNNMusicDataSetPreparer,
    load_midi_objs,
    load_nottingham_objs,
    load_just_that_one_test_song,
    download_and_save_data,
)
from src.utils.visualization import plot_piano_roll, save_audio_file
from datetime import datetime
from src.models import *
from src.prediction import *


class RNNMusicExperiment:
    """
    Super class for all experiments
    """

    def __init__(
        self,
        sequence_length=15,
        epochs=10,
        learning_rate=0.001,
        batch_size=64,
        num_music_files=2,
        vocab_size=128,
    ) -> None:
        self.common_config = {
            "seq_length": sequence_length,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "num_music_files": num_music_files,
            "vocab_size": vocab_size,
        }
        return

    def get_model(self):
        return self.model

    def get_history(self):
        return self.history

    def get_name(self):
        raise NotImplementedError

    def set_model(self):
        raise NotImplementedError

    def run(self):
        self.set_model()
        self.loaded_data = self.load_data()
        self.prepared_data = self.prepare_data(self.loaded_data)
        print("Training...")
        self.model, self.history = self.train_model(self.prepared_data)
        # Save training stats?
        # Pickle the model?

        print("Predicting data...")
        self.predicted, self.probs = self.predict_data(self.model, self.loaded_data)

        print("Saving data...")
        self.plot_and_save_predicted_data(self.predicted)
        self.plot_and_save_predicted_data(self.probs)
        self.create_and_save_predicted_audio(self.predicted, "_music_")
        # Save music file?
        # Save music output plot?

    def basic_load_data(self):
        loaded = load_midi_objs(
            num_files=self.common_config["num_music_files"],
            seq_length=self.common_config["seq_length"],
        )
        return loaded

    def load_data(self):
        raise NotImplementedError

    def prepare_data(self):
        # TODO: Can the be in the form seq_ds or (X_train, y_train)
        raise NotImplementedError

    def train_model(self, prepared_data):
        model, callbacks = self.model, self.callbacks
        print(f"type prepared_data is {type(prepared_data)}")
        # seq_length, _ = song_df.shape
        # buffer_size = n_notes - seq_length  # the number of items in the dataset
        buffer_size = 100
        train_ds = (
            prepared_data.shuffle(buffer_size)
            .batch(self.common_config["batch_size"], drop_remainder=True)
            .cache()
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        history = model.fit(
            train_ds,
            epochs=self.common_config["epochs"],
            callbacks=callbacks,
        )
        return model, history

    def predict_data(self):
        raise NotImplementedError

    def get_save_plot_path(self, str_ind=""):
        out = "plots/"
        out += self.get_name()
        out += str_ind
        out += "_".join(
            [str(x).replace(".", "dot") for x in self.common_config.values()]
        )
        out += "__"
        out += datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        out += ".png"
        return out

    def get_save_audio_path(self, str_ind=""):
        out = "audio/"
        out += self.get_name()
        out += str_ind
        out += "_".join(
            [str(x).replace(".", "dot") for x in self.common_config.values()]
        )
        out += datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        out += ".wav"
        return out

    def plot_and_save_predicted_data(self, predicted, str_ind=""):
        if predicted is not None:
            plot_piano_roll(
                predicted, self.get_save_plot_path("_both_" + str_ind), plot_type="both"
            )
            plot_piano_roll(
                predicted,
                self.get_save_plot_path("_artic_" + str_ind),
                plot_type="artic",
            )
            plot_piano_roll(
                predicted,
                self.get_save_plot_path("_note_hold_" + str_ind),
                plot_type="note_hold",
            )

    def create_and_save_predicted_audio(self, predicted, str_ind=""):
        save_audio_file(
            predicted, self.get_save_audio_path("_artic_" + str_ind), audio_type="artic"
        )
        save_audio_file(
            predicted,
            self.get_save_audio_path("_note_hold_" + str_ind),
            audio_type="note_hold",
        )


class RNNMusicExperimentTen(RNNMusicExperiment):
    """Exp using time / note network

    Args:
        RNNMusicExperimentTen ([type]): [description]
    """

    def __init__(self, *args, dropout=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.common_config["dropout"] = dropout

    def get_name(self):
        return "Exp10"

    def run(self):
        self.set_model()
        loaded_midi = self.load_data()
        self.prepared_data = self.prepare_data(loaded_midi)
        print("Training...")
        self.model, self.history = self.train_model(self.prepared_data)
        self.model.save(
            "/Users/trevorgordon/Library/Mobile Documents/com~apple~CloudDocs/Documents/root/Columbia/Spring2022/NNDL/music_generation_rnn/models/latest"
        )
        self.predict_and_save_data()

    def predict_and_save_data(self, str_id=""):

        print("Predicting data...")
        self.predicted, self.probs = self.predict_data(self.model, self.prepared_data)
        print("Saving data...")
        self.plot_and_save_predicted_data(self.predicted, str_id + "_predicted_")
        self.plot_and_save_predicted_data(self.probs, str_id + "_probs_")
        self.create_and_save_predicted_audio(self.predicted, str_id + "_music_")
        # Save music file?
        # Save music output plot?

    def prepare_data(self, loaded_data):
        # seq_ds is in form X, y here
        X, y = MidiSupport().prepare_song_note_invariant_plus_beats_and_more(
            loaded_data, vicinity=24
        )
        self.X_before_window = X
        seq_ds = MidiSupport().prepare_windowed_for_note_time_invariant(
            X, seq_length=128
        )
        seq_ds.shuffle(buffer_size=10)
        return seq_ds

    def predict_data(self, model, prepared_data):
        return predict_time_note_model(
            model, self.X_before_window, dropout=self.common_config["dropout"], size=10
        )

    def set_model(self):
        self.model = NoteTimeMusicPredictionRNNShallow(
            learning_rate=0.001,
            dropout=self.common_config["dropout"],
        )
        self.callbacks = haltCallback(0.4)

    def load_data(self):
        loaded = load_just_that_one_test_song(
            num_files=1,
            seq_length=128,
        )
        return loaded

    def train_model(self, prepared_data):
        model, callbacks = self.model, self.callbacks

        history = model.fit(
            prepared_data,
            epochs=10,
            callbacks=callbacks,
        )
        return model, history


class RNNMusicExperimentEleven(RNNMusicExperimentTen):
    """Same as 10 but training on multiple songs

    Args:
        RNNMusicExperimentEleven ([type]): [description]
    """

    def get_name(self):
        return "Exp11"

    def load_data(self):
        print("loading nottingham")
        loaded = load_nottingham_objs(
            num_files=self.common_config["num_music_files"],
            seq_length=self.common_config["seq_length"],
        )
        return loaded
