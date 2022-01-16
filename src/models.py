import tensorflow as tf


class SwapLayer(tf.keras.layers.Layer):
    """Tensorflow layer that swaps the 1st and 2nd dimension
    """
    def __init__(self):
        super(SwapLayer, self).__init__()

    def call(self, orig_tensor, training=None):
        swapped = tf.transpose(orig_tensor, [1, 0, 2])
        return swapped


class NoteTimeMusicPredictionRNN:
    """Tensorflow RNN model for music prediction

    This class wraps around a tensorflow model to provide some methods for predict part of the network

    Description of the model is available here:
    https://trevor16gordon.github.io/notes/music_gen_rnn.html

    This model requires the time sequence length to be 128 so that after swapping the time and note
    axis we still have the same model dimensions. Our output sequence length must be 128 for the 128 midi notes.
    The output shape is (128, 128, 2) as there is a note hold and a note articulation per note.

    input_shape_a = (128, 128, total_vicinity)
    input_shape_b = (128, 128, 2)

    """
    def __init__(self, learning_rate=0.001, total_vicinity=53, dropout=0) -> None:
        self.build_model(
            learning_rate=learning_rate, total_vicinity=total_vicinity, dropout=dropout
        )

    def build_model(self, learning_rate=0.001, total_vicinity=53, dropout=0):
        """Build the tf model

        Args:
            learning_rate (float, optional): Defaults to 0.001.
            total_vicinity (int, optional): Defaults to 53.
            dropout (int, optional): Defaults to 0.
        """

        # Batch size and temporal sequence length need to be 128 so they can be swapped
        # and the output sequence length is 128
        temporal_seq_length = 128

        input_shape = (temporal_seq_length, total_vicinity)
        input_shape_b = (temporal_seq_length, 2)

        input_a = tf.keras.Input(shape=input_shape, name="input_a")
        input_b = tf.keras.Input(shape=input_shape_b, name="input_b")

        # Time layers: recursion along the time dimension of music
        mod_time = SwapLayer()(input_a)
        mod_time = tf.keras.layers.LSTM(200, return_sequences=True, dropout=dropout)(
            mod_time
        )
        mod_time = tf.keras.layers.LSTM(200, return_sequences=True, dropout=dropout)(
            mod_time
        )
        mod_time = SwapLayer()(mod_time)
        mod_time = tf.keras.Model(inputs=input_a, outputs=mod_time)

        # Note layers take in the previously selected note along with time layer output to predict
        combined = tf.keras.layers.concatenate([mod_time.output, input_b])

        # Note layers: recursion along the note dimension of music
        mod_notes = tf.keras.layers.LSTM(100, return_sequences=True)(combined)
        mod_notes = tf.keras.layers.LSTM(50, return_sequences=True)(mod_notes)
        mod_notes = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(2, activation="sigmoid")
        )(mod_notes)

        model = tf.keras.Model(inputs=[input_a, input_b], outputs=[mod_notes])
        model.summary()

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        model.compile(
            loss=loss,
            optimizer=optimizer,
            # metrics=[tf.keras.metrics.BinaryCrossentropy()
            metrics=["mse"],
        )
        self.model = model

    def fit(self, *args, **kwargs):
        """Pass through to tf.model.fit
        """
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """Pass through to tf.model.predict
        """
        return self.model.predict(*args, **kwargs)

    def save(self, *args, **kwargs):
        """Pass through to tf.model.save
        """
        return self.model.save(*args, **kwargs)

    def predict_time_lstm_output(self, *args, **kwargs):
        """Prediction that gives the output of the time axis layers.

        This is useful for prediction because 1 time axis prediction will be reused for
        128 note axis prediction outputs

        Returns:
            tf.tensor: hidden layer output
        """
        intermediate_layer_model = tf.keras.Model(
            inputs=self.model.input, outputs=self.model.layers[3].output
        )
        intermediate_output = intermediate_layer_model.predict(*args, **kwargs)
        return intermediate_output

    def create_sub_model(self):
        """Create a tf sub model that predicts starting at the note axis layers.

        This is useful for prediction because 1 time axis prediction will be reused for
        128 note axis prediction outputs
        """
        input_shape = (128, 200)
        new_input = tf.keras.Input(shape=input_shape, name="new_input")
        old_input_b = self.model.layers[5]
        concat_lay = self.model.layers[6]([new_input, old_input_b.output])
        mod_notes = self.model.layers[7](concat_lay)
        mod_notes = self.model.layers[8](mod_notes)
        mod_notes = self.model.layers[9](mod_notes)
        sub_model = tf.keras.Model(
            inputs=[new_input, old_input_b.input], outputs=[mod_notes]
        )
        self.sub_model = sub_model

    def predict_sub_model(self, *args, **kwargs):
        """Predict from sub model

        This is useful for prediction because 1 time axis prediction will be reused for
        128 note axis prediction outputs
        """
        if not self.sub_model:
            raise ValueError(
                "self.sub_model doesn't exist yet. Please call create_sub_model first!"
            )
        return self.sub_model.predict(*args, **kwargs)


class NoteTimeMusicPredictionRNNShallow:
    """Like NoteTimeMusicPredictionRNN but with only a single
    LSTM per note / time layer

    """
    def __init__(self, learning_rate=0.001, total_vicinity=53, dropout=0) -> None:
        self.build_model(
            learning_rate=learning_rate, total_vicinity=total_vicinity, dropout=dropout
        )

    def build_model(self, learning_rate=0.001, total_vicinity=53, dropout=0):
        """Build the tf model

        Args:
            learning_rate (float, optional): Defaults to 0.001.
            total_vicinity (int, optional): Defaults to 53.
            dropout (int, optional): Defaults to 0.
        """

        # Batch size and temporal sequence length need to be 128 so they can be swapped
        # and the output sequence length is 128
        temporal_seq_length = 128

        input_shape = (temporal_seq_length, total_vicinity)
        input_shape_b = (temporal_seq_length, 2)

        input_a = tf.keras.Input(shape=input_shape, name="input_a")
        input_b = tf.keras.Input(shape=input_shape_b, name="input_b")

        # Time layers: recursion along the time dimension of music
        mod_time = SwapLayer()(input_a)
        mod_time = tf.keras.layers.LSTM(200, return_sequences=True, dropout=dropout)(
            mod_time
        )
        mod_time = SwapLayer()(mod_time)
        mod_time = tf.keras.Model(inputs=input_a, outputs=mod_time)

        # Note layers take in the previously selected note along with time layer output to predict
        combined = tf.keras.layers.concatenate([mod_time.output, input_b])

        # Note layers: recursion along the note dimension of music
        mod_notes = tf.keras.layers.LSTM(100, return_sequences=True)(combined)
        mod_notes = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(2, activation="sigmoid")
        )(mod_notes)

        model = tf.keras.Model(inputs=[input_a, input_b], outputs=[mod_notes])
        model.summary()

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        model.compile(
            loss=loss,
            optimizer=optimizer,
            # metrics=[tf.keras.metrics.BinaryCrossentropy()
            metrics=["mse"],
        )
        self.model = model

    def fit(self, *args, **kwargs):
        """Pass through to tf.model.fit
        """
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """Pass through to tf.model.predict
        """
        return self.model.predict(*args, **kwargs)

    def save(self, *args, **kwargs):
        """Pass through to tf.model.save
        """
        return self.model.save(*args, **kwargs)

    def predict_time_lstm_output(self, *args, **kwargs):
        """Prediction that gives the output of the time axis layers.

        This is useful for prediction because 1 time axis prediction will be reused for
        128 note axis prediction outputs

        Returns:
            tf.tensor: hidden layer output
        """
        intermediate_layer_model = tf.keras.Model(
            inputs=self.model.input, outputs=self.model.layers[2].output
        )
        intermediate_output = intermediate_layer_model.predict(*args, **kwargs)
        return intermediate_output

    def create_sub_model(self):
        """Create a tf sub model that predicts starting at the note axis layers.

        This is useful for prediction because 1 time axis prediction will be reused for
        128 note axis prediction outputs
        """
        input_shape = (128, 200)
        new_input = tf.keras.Input(shape=input_shape, name="new_input")
        old_input_b = self.model.layers[4]
        concat_lay = self.model.layers[5]([new_input, old_input_b.output])
        mod_notes = self.model.layers[6](concat_lay)
        mod_notes = self.model.layers[7](mod_notes)
        sub_model = tf.keras.Model(
            inputs=[new_input, old_input_b.input], outputs=[mod_notes]
        )
        self.sub_model = sub_model

    def predict_sub_model(self, *args, **kwargs):
        """Predict from sub model

        This is useful for prediction because 1 time axis prediction will be reused for
        128 note axis prediction outputs
        """
        if not self.sub_model:
            raise ValueError(
                "self.sub_model doesn't exist yet. Please call create_sub_model first!"
            )
        return self.sub_model.predict(*args, **kwargs)


class haltCallback(tf.keras.callbacks.Callback):
    """TF callback to stop model training when a specific loss value is reached

    Args:
        tf ([type]): [description]
    """

    def __init__(self, halt_loss_val) -> None:
        super().__init__()
        self.halt_loss_val = halt_loss_val

    def on_batch_end(self, batch, logs={}):
        if logs.get("loss") <= self.halt_loss_val:
            print(
                f"\n\n\nReached {self.halt_loss_val} loss value so cancelling training!\n\n\n"
            )
            self.model.stop_training = True
