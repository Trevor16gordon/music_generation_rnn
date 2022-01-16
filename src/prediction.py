import numpy as np
import pandas as pd
from src.data_preparation import MusicDataPreparer
from src.models import NoteTimeMusicPredictionRNN, NoteTimeMusicPredictionRNNShallow


def predict_time_note_model(model, prepared_training_data, size=16, dropout=0):
    """Predict music sequence.

    This function starts with training data output from MusicDataPreparer().prepare_song_note_invariant_plus_beats_and_more.

    - Randomly select a 128 music sequence length from prepared_training_data
    - For prediction in size:
        - Predict time, note element
        - Grab intermediate_output and setup sub_model for prediction
        - For note in remaining 127 notes
            - Predict next note probabilities using intermediate_output, sub_model 
              and previous notes in this timestep
            - Randomly select if note should be played/articulated using np.random.binomial
        - Reformat all 128 notes into the input shape by convolving across note dimension to create window etc.

    Args:
        model (NoteTimeMusicPredictionRNN): TF model for prediction
        prepared_training_data (np.array): Prepared data according to MusicDataPreparer().prepare_song_note_invariant_plus_beats_and_more
        size (int, optional): Output prediction sequence length. Defaults to 16.
        dropout (float, optional): Training dropout to account for in output probabilities. Defaults to 0.

    Raises:
        TypeError: If the model passed is of the wrong type
        ValueError: If the shape of the input training data is too short

    Returns:
        np.array: Predicted output notes as binary value piano roll
        np.array: Predicted output probability as floats piano roll
    """

    num_beats = 128
    num_notes = 128
    total_vicinity = 53
    note_vicinity = 24

    if not isinstance(model,(NoteTimeMusicPredictionRNN, NoteTimeMusicPredictionRNNShallow)):
        raise TypeError(
            "Model passed in needs to be of type NoteTimeMusicPredictionRNN"
        )

    if prepared_training_data.shape[0] < num_beats:
        raise ValueError(
            f"prepared_training_data.shape[0] needs to be > num_beats for prediction"
        )
    reshaped_train_data = prepared_training_data

    outputs = []
    all_probs = []

    offset = np.random.choice(range(len(reshaped_train_data) - num_beats))
    # offset = 0

    input_notes_reshape = reshaped_train_data[offset + 1 : offset + 1 + num_beats, :, :]
    input_notes_b_reshape = reshaped_train_data[offset : offset + num_beats, :, 12:14]

    last_beats = [str(x) for x in input_notes_reshape[-1, -1, -4:]]
    last_beats_int = int("".join(last_beats), 2)

    for l in range(size):
        print(f"{l}/{size}")
        out_arr = np.zeros((num_notes, 2))
        probs_arr = np.zeros((num_notes, 2))

        for i in range(num_notes):

            if i == 0:
                probs = model.predict(
                    [input_notes_reshape, input_notes_b_reshape], batch_size=128
                )

                # Sub Model
                intermediate_output = model.predict_time_lstm_output(
                    [input_notes_reshape, input_notes_b_reshape], batch_size=128
                )
                model.create_sub_model()

            else:
                probs = model.predict_sub_model(
                    [intermediate_output[-1:, :, :], input_notes_b_reshape[-1:, :, :]],
                    batch_size=128,
                )

            last_new_play_artic_probs = probs[-1, i, :]
            last_new_play_artic_probs = last_new_play_artic_probs * (1 - dropout + 0.3)
            last_new_play_artic = np.random.binomial(
                1, last_new_play_artic_probs, size=None
            ).reshape((1, 2))

            if i < (num_notes - 1):
                input_notes_b_reshape[-1, i : i + 1, :] = last_new_play_artic
            else:
                input_notes_b_reshape = np.concatenate(
                    [input_notes_b_reshape[1:, :, :], np.zeros((1, num_beats, 2))],
                    axis=0,
                )

            probs_arr[i, 0] = last_new_play_artic_probs[0]
            probs_arr[i, 1] = last_new_play_artic_probs[1]
            out_arr[i, 0] = last_new_play_artic[0, 0]
            out_arr[i, 1] = last_new_play_artic[0, 1]

        out = out_arr.reshape((256, 1))
        probs = probs_arr.reshape((256, 1))

        # Need to window across out to get the input form again.
        out = out.reshape((256, 1))
        probs = probs.reshape((256,))
        outputs.append(
            out.reshape(
                256,
            )
        )
        all_probs.append(probs)

        # note_vicinity = total_vicinity-4-12-12-1
        next_pred, _, _ = MusicDataPreparer().windowed_data_across_notes_time(
            out, mask_length_x=note_vicinity, return_labels=False
        )  # Return (total_vicinity, 128)

        # Get array of Midi values for each note value
        n_notes = 128
        midi_row = MusicDataPreparer().add_midi_value(next_pred, n_notes)

        # Get array of one hot encoded pitch values for each note value
        pitchclass_rows = MusicDataPreparer().calculate_pitchclass(midi_row, next_pred)

        # Add total_pitch count repeated for each note window
        previous_context = MusicDataPreparer().build_context(
            next_pred, midi_row, pitchclass_rows
        )

        midi_row = midi_row.reshape((1, -1))
        next_pred = np.vstack((next_pred, midi_row, pitchclass_rows, previous_context))

        last_beats_int += 1
        last_beats_int = last_beats_int % 16
        next_beats_ind = np.array([int(x) for x in bin(last_beats_int)[2:].zfill(4)])
        next_beats_ind = next_beats_ind.reshape((4, 1))
        next_beats_ind = np.repeat(next_beats_ind, num_notes, axis=1)

        # TODO, check if beat is correctly increasing: might need to flip it before adding
        last_new_note = np.concatenate([next_pred, next_beats_ind])
        last_new_note = last_new_note[np.newaxis, :, :]  # Shape now  (1, 28, 128)
        last_new_note = np.swapaxes(last_new_note, 1, 2)  # Shape now  (1, 128, 28)
        # last_new_note = np.swapaxes(last_new_note, 0, 1) # Shape now  (128, 1, 28)

        together = np.concatenate(
            [input_notes_reshape[1:, :, :], last_new_note], axis=0
        )
        input_notes_reshape = together

    outputs_joined = pd.DataFrame(outputs)
    all_probs_joined = pd.DataFrame(all_probs)
    return outputs_joined, all_probs_joined
