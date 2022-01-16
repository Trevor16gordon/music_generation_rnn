from src.experiments import RNNMusicExperiment

class RNNMusicExperimentOne(RNNMusicExperiment):
    def set_model(self):
        self.model, self.callbacks = model_5_lstm_layer_with_artic(
            learning_rate=self.common_config["learning_rate"],
            seq_length=self.common_config["seq_length"],
        )

    def get_name(self):
        return "Exp1"

    def load_data(self):
        return self.basic_load_data()

    def prepare_data(self, midi_objs):
        play_articulated = MusicDataPreparer().all_midi_obj_to_play_articulate(midi_objs)
        seq_ds = RNNMusicDataSetPreparer().prepare(
            play_articulated.T, seq_length=self.common_config["seq_length"]
        )
        # TODO: Some models return a DataSet and some return X_train, y_train
        return seq_ds

    def predict_data(self, model, loaded_data):
        return predict_notes_256_sigmoid(
            model=model, train_data=loaded_data, size=self.common_config["seq_length"]
        )


class RNNMusicExperimentTwo(RNNMusicExperimentOne):
    """Builds on Exp 1 by adding limited connectivity

    Args:
        RNNMusicExperiment ([type]): [description]
    """

    def get_name(self):
        return "Exp2"

    def set_model(self):
        self.model, self.callbacks = model_4_lstm_layer_limited_connectivity(
            learning_rate=self.common_config["learning_rate"],
            seq_length=self.common_config["seq_length"],
        )


class RNNMusicExperimentThree(RNNMusicExperiment):
    """Note invariance with articularion

    Args:
        RNNMusicExperiment ([type]): [description]
    """

    def run(self):
        self.set_model()
        loaded_midi = self.load_data()
        self.prepared_data = self.prepare_data(loaded_midi)
        print("Training...")
        self.model, self.history = self.train_model(self.prepared_data)
        self.predict_and_save_data()

    def predict_and_save_data(self, str_id=""):

        print("Predicting data...")
        predicted, probs = self.predict_data(self.model, self.prepared_data)
        print("Saving data...")
        self.plot_and_save_predicted_data(predicted, str_id + "_predicted_")
        self.plot_and_save_predicted_data(probs, str_id + "_probs_")
        self.create_and_save_predicted_audio(predicted, str_id + "_music_")
        # Save music file?
        # Save music output plot?

    def get_name(self):
        return "Exp3"

    def set_model(self):
        self.model, self.callbacks = model_6_note_invariant(
            learning_rate=self.common_config["learning_rate"]
        )

    def train_model(self, prepared_data):
        """Train model overwrite

        Args:
            prepared_data (tuple): X, y

        """
        model, callbacks = self.model, self.callbacks
        history = model.fit(
            prepared_data[0],
            prepared_data[1],
            epochs=self.common_config["epochs"],
            callbacks=callbacks,
            batch_size=1,
        )
        return model, history

    def load_data(self):
        return self.basic_load_data()

    def prepare_data(self, loaded_data):
        # seq_ds is in form X, y here
        seq_ds = MusicDataPreparer().prepare_song_note_invariant_plus_beats(loaded_data)
        # TODO: Some models return a DataSet and some return X_train, y_train
        return seq_ds

    def predict_data(self, model, prepared_data):
        return predict_notes_note_invariant(model, prepared_data[0], size=200)


class RNNMusicExperimentFour(RNNMusicExperiment):
    """Note invariance with articularion and beats and extras


    Args:
        RNNMusicExperiment ([type]): [description]
    """

    def __init__(self, *args, num_beats_for_prediction=1, note_vicinity=24, **kwargs):
        super().__init__(*args, **kwargs)
        self.common_config["num_beats_for_prediction"] = num_beats_for_prediction
        self.common_config["note_vicinity"] = note_vicinity

    def get_name(self):
        return "Exp4"

    def set_model(self):
        self.model, self.callbacks = model_6_note_invariant(
            learning_rate=self.common_config["learning_rate"],
            # Total vicinity 24 notes + 4 beats + 1 midi + 12 context + 12 pitchclass
            total_vicinity=self.common_config["note_vicinity"] + 4 + 1 + 12 + 12,
        )

    def run(self):
        self.set_model()
        loaded_midi = self.load_data()
        self.prepared_data = self.prepare_data(loaded_midi)
        print("Training...")
        self.model, self.history = self.train_model(self.prepared_data)
        self.predict_and_save_data()

    def predict_and_save_data(self, str_id=""):

        print("Predicting data...")
        predicted, probs = self.predict_data(self.model, self.prepared_data)
        print("Saving data...")
        self.plot_and_save_predicted_data(predicted, str_id + "_predicted_")
        self.plot_and_save_predicted_data(probs, str_id + "_probs_")
        self.create_and_save_predicted_audio(predicted, str_id + "_music_")
        # Save music file?
        # Save music output plot?

    def train_model(self, prepared_data):
        model, callbacks = self.model, self.callbacks
        history = model.fit(
            prepared_data[0],
            prepared_data[1],
            epochs=self.common_config["epochs"],
            callbacks=callbacks,
            batch_size=self.common_config["batch_size"],
        )
        return model, history

    def load_data(self):
        return self.basic_load_data()

    def prepare_data(self, loaded_data):
        # seq_ds is in form X, y here
        seq_ds = MusicDataPreparer().prepare_song_note_invariant_plus_beats_and_more(
            loaded_data
        )
        # TODO: Some models return a DataSet and some return X_train, y_train
        return seq_ds

    def predict_data(self, model, prepared_data):
        return predict_notes_note_invariant_plus_extras_multiple_time_steps(
            model,
            prepared_data[0],
            size=200,
            num_beats=self.common_config["num_beats_for_prediction"],
        )


class RNNMusicExperimentFive(RNNMusicExperimentFour):
    """Note invariance with articularion and beats and extras


    Same as experiment Four but just run on a single song

    Args:
        RNNMusicExperiment ([type]): [description]
    """

    def get_name(self):
        return "Exp5"

    def basic_load_data(self):
        loaded = load_just_that_one_test_song(
            num_files=self.common_config["num_music_files"],
            seq_length=self.common_config["seq_length"],
        )
        return loaded


class RNNMusicExperimentSeven(RNNMusicExperimentFive):
    """Study on number of notes to have in the vicinity

    Args:
        RNNMusicExperimentFour ([type]): [description]
    """

    def get_name(self):
        return "Exp7"

    def prepare_data(self, loaded_data):
        # seq_ds is in form X, y here
        seq_ds = MusicDataPreparer().prepare_song_note_invariant_plus_beats_and_more(
            loaded_data, vicinity=self.common_config["note_vicinity"]
        )
        # TODO: Some models return a DataSet and some return X_train, y_train
        return seq_ds

    def predict_data(self, model, prepared_data):
        return predict_notes_note_invariant_plus_extras_multiple_time_steps(
            model,
            prepared_data[0],
            size=200,
            note_vicinity=self.common_config["note_vicinity"],
            num_beats=self.common_config["num_beats_for_prediction"],
        )


class RNNMusicExperimentEight(RNNMusicExperimentSeven):
    """Study on increasing model capacitty

    Args:
        RNNMusicExperimentFour ([type]): [description]
    """

    def __init__(self, *args, num_hidden_nodes=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.common_config["num_hidden_nodes"] = num_hidden_nodes

    def get_name(self):
        return "Exp8"

    def set_model(self):
        self.model, self.callbacks = model_6_note_invariant(
            learning_rate=self.common_config["learning_rate"],
            num_hidden_nodes=self.common_config["num_hidden_nodes"],
            # Total vicinity 24 notes + 4 beats + 1 midi + 12 context + 12 pitchclass
            total_vicinity=self.common_config["note_vicinity"] + 4 + 1 + 12 + 12,
        )


class RNNMusicExperimentNine(RNNMusicExperimentEight):
    """Build on Experiment Eight but trys to train on many songs

    Args:
        RNNMusicExperimentEight ([type]): [description]
    """

    def basic_load_data(self):
        loaded = load_midi_objs(
            num_files=self.common_config["num_music_files"],
            seq_length=self.common_config["seq_length"],
        )
        return loaded


class RNNMusicExperimentTFRef(RNNMusicExperiment):
    """Google Tutorial Version
    RNNMusicExperiment ([type]): Implements the model described by Google here. Only used for performance
    comparisons: https://www.tensorflow.org/tutorials/audio/music_generation
    """

    def get_name(self):
        return "ExpTFRef"

    def set_model(self):
        print(f"in get_model self is {self}")
        self.model, self.callbacks = model_7_google(
            learning_rate=self.common_config["learning_rate"],
            seq_length=self.common_config["seq_length"],
        )

    def run(self):
        self.key_order = ["pitch", "step", "duration"]
        super().run()

    def load_data(self):
        return self.basic_load_data()

    def prepare_data(self, loaded_data):
        # seq_ds is in form X, y here
        seq_length = self.common_config["seq_length"]
        num_files = self.common_config["num_music_files"]
        batch_size = self.common_config["batch_size"]

        all_notes = []
        for f in loaded_data[:num_files]:
            notes = MusicDataPreparer().midi_to_notes(f)
            all_notes.append(notes)

        all_notes = pd.concat(all_notes)
        self.all_notes = all_notes
        n_notes = len(all_notes)

        train_notes = np.stack([all_notes[key] for key in self.key_order], axis=1)

        notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)

        seq_ds = RNNMusicDataSetPreparer().create_sequences(
            notes_ds, seq_length=seq_length, key_order=self.key_order
        )

        buffer_size = n_notes - seq_length  # the number of items in the dataset
        train_ds = (
            seq_ds.shuffle(buffer_size)
            .batch(batch_size, drop_remainder=True)
            .cache()
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        return train_ds

    def train_model(self, prepared_data):
        """Train model overwrite

        Args:
            prepared_data (Tensorflow dataset): prepared_data

        """
        model, callbacks = self.model, self.callbacks
        history = model.fit(
            prepared_data,
            epochs=self.common_config["epochs"],
            callbacks=callbacks,
        )
        return model, history

    def predict_data(self, model, prepared_data):
        key_order = self.key_order
        all_notes = self.all_notes
        seq_length = self.common_config["seq_length"]
        vocab_size = self.common_config["vocab_size"]

        def predict_next_note(
            notes: np.ndarray, keras_model: tf.keras.Model, temperature: float = 1.0
        ) -> int:
            """Generates a note IDs using a trained sequence model."""

            assert temperature > 0

            # Add batch dimension
            inputs = tf.expand_dims(notes, 0)

            predictions = model.predict(inputs)
            pitch_logits = predictions["pitch"]
            step = predictions["step"]
            duration = predictions["duration"]

            pitch_logits /= temperature
            pitch = tf.random.categorical(pitch_logits, num_samples=1)
            pitch = tf.squeeze(pitch, axis=-1)
            duration = tf.squeeze(duration, axis=-1)
            step = tf.squeeze(step, axis=-1)

            # `step` and `duration` values should be non-negative
            step = tf.maximum(0, step)
            duration = tf.maximum(0, duration)

            return int(pitch), float(step), float(duration)

        temperature = 2.0
        num_predictions = 120

        sample_notes = np.stack([all_notes[key] for key in key_order], axis=1)

        # The initial sequence of notes; pitch is normalized similar to training
        # sequences
        input_notes = sample_notes[:seq_length] / np.array([vocab_size, 1, 1])

        generated_notes = []
        prev_start = 0
        for _ in range(num_predictions):
            pitch, step, duration = predict_next_note(input_notes, model, temperature)
            start = prev_start + step
            end = start + duration
            input_note = (pitch, step, duration)
            generated_notes.append((*input_note, start, end))
            input_notes = np.delete(input_notes, 0, axis=0)
            input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
            prev_start = start

        generated_notes = pd.DataFrame(
            generated_notes, columns=(*self.key_order, "start", "end")
        )
        return generated_notes, None
