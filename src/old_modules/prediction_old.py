import numpy as np

def predict_notes_256_sigmoid(model, train_data, size=10):
    """Predict notes

    Expects model predictioin to be sigmoid output 0 or 1
    Plays notes if probability is greater than 0.5

    Args:
        model (tf.keras.model): The model
            Expects the model to give output in dictionary that has "pitch"
        size (int): Number of notes to predict Defaults to 10.
        training data pd.DataFrame: Training data to pull random data from for start of prediction

    Returns:
        pd.DataFrame: The predicted notes as shape (128 x size)
    """
    outputs = []
    all_probs = []
    offset = np.random.choice(range(len(train_data) - size))
    # offset= 1233
    input_notes = train_data.iloc[offset : offset + size]
    input_notes_reshape = input_notes.values.reshape((1, size, 260))
    last_beats = [str(x) for x in input_notes_reshape[:, -1, -4:][0]]
    last_beats_int = int("".join(last_beats), 2)
    for l in range(size):
        probs = model.predict(input_notes_reshape)["pitch"].flatten()
        # probs = probs/probs.sum()
        # selected = np.random.choice(range(len(probs)), p=probs, size=2)
        out = np.zeros_like(probs)
        # out[selected] = 1
        out[probs > 0.5] = 1
        outputs.append(out)
        all_probs.append(probs)

        last_beats_int += 1
        last_beats_int = last_beats_int % 16
        next_beats_ind = [int(x) for x in bin(last_beats_int)[2:].zfill(4)]

        last_new_note = np.concatenate([out, next_beats_ind]).reshape((1, 1, 260))
        input_notes_reshape = np.concatenate(
            [input_notes_reshape[:, 1:, :], last_new_note], axis=1
        )

    return pd.DataFrame(outputs), pd.DataFrame(all_probs)


def predict_notes_note_invariant(model, reshaped_train_data, size=10):
    """Predict notes

    Expects model predictioin to be sigmoid output 0 or 1
    Expects model that predict sequence of 128 at a time which amount to "the next note"
    Plays notes if probability is greater than 0.5

    Args:
        reshaped_train_data: Needs to already be X_prepared unreavelled form
        model (tf.keras.model): The model
            Expects the model to give output in dictionary that has "pitch"
        size (int): Number of notes to predict Defaults to 10.
        training data pd.DataFrame: Training data to pull random data from for start of prediction

    Returns:
        pd.DataFrame: The predicted notes as shape (128 x size)
    """
    outputs = []
    all_probs = []
    num_notes = 128
    num_beats = 1
    elements_per_time_step = 128
    offset = np.random.choice(range(len(reshaped_train_data)))
    input_notes = reshaped_train_data[offset : offset + num_beats, :, :]
    # input_notes_reshape = input_notes.reshape(1, num_notes, total_vicinity)
    input_notes_reshape = input_notes
    last_beats = [str(x) for x in input_notes_reshape[0, -1, -4:]]
    last_beats_int = int("".join(last_beats), 2)
    for l in range(size):

        # probs = model.predict(input_notes_reshape)["pitch"].flatten()
        # *******
        probs = model.predict(input_notes_reshape)
        # probs shape should be something like (256, beats)
        probs = pd.DataFrame(
            probs.reshape(num_beats, elements_per_time_step * 2, order="C").T
        )

        # probs = probs[:, -1]
        # output sequence will be the same length as the input. Try to either take the first or the last beat

        play_bias = 0.01
        probs = probs + play_bias
        probs[probs > 1] = 1
        probs[probs < 0] = 0
        out = np.zeros_like(probs)
        # out[tst_output_reshaped > 0.5 ] = 1

        # This part fails when we have multiple output.
        out = np.random.binomial(1, probs, size=None).reshape(num_notes * 2)

        # Need to window across out to get the input form again.
        out = out.reshape((256, 1))
        probs = probs.values.reshape((256,))

        outputs.append(
            out.reshape(
                256,
            )
        )
        all_probs.append(probs)

        next_pred, _, _ = MidiSupport().windowed_data_across_notes_time(
            out, mask_length_x=24, return_labels=False
        )  # Return (24, 128)

        # *******

        # pdb.set_trace()
        last_beats_int += 1
        last_beats_int = last_beats_int % 16
        next_beats_ind = np.array([int(x) for x in bin(last_beats_int)[2:].zfill(4)])
        next_beats_ind = next_beats_ind.reshape((4, 1))
        next_beats_ind = np.repeat(next_beats_ind, num_notes, axis=1)

        # TODO, check if beat is correctly increasing: might need to flip it before adding
        last_new_note = np.concatenate([next_pred, next_beats_ind])
        last_new_note = last_new_note[np.newaxis, :, :]  # Shape now  (1, 28, 128)
        last_new_note = np.swapaxes(last_new_note, 1, 2)  # Shape now  (1, 128, 28)
        input_notes_reshape = last_new_note

    outputs_joined = pd.DataFrame(outputs)
    all_probs_joined = pd.DataFrame(all_probs)
    return outputs_joined, all_probs_joined


def predict_notes_note_invariant_plus_extras(model, reshaped_train_data, size=10):
    """Predict notes

    Expects model predictioin to be sigmoid output 0 or 1
    Expects model that predict sequence of 128 at a time which amount to "the next note"
    Plays notes if probability is greater than 0.5

    Args:
        reshaped_train_data: Needs to already be X_prepared unreavelled form
        model (tf.keras.model): The model
            Expects the model to give output in dictionary that has "pitch"
        size (int): Number of notes to predict Defaults to 10.
        training data pd.DataFrame: Training data to pull random data from for start of prediction

    Returns:
        pd.DataFrame: The predicted notes as shape (128 x size)
    """
    outputs = []
    all_probs = []
    num_notes = 128
    elements_per_time_step = 128
    num_beats = 1
    offset = np.random.choice(range(len(reshaped_train_data)))
    input_notes = reshaped_train_data[offset : offset + num_beats, :, :]
    # input_notes_reshape = input_notes.reshape(1, num_notes, total_vicinity)
    input_notes_reshape = input_notes
    last_beats = [str(x) for x in input_notes_reshape[0, -1, -4:]]
    last_beats_int = int("".join(last_beats), 2)

    last_beats_int = 0

    for l in range(size):

        # probs = model.predict(input_notes_reshape)["pitch"].flatten()
        # *******
        # model.reset_states()
        probs = model.predict(input_notes_reshape)

        # probs shape should be something like (256, beats)
        probs = pd.DataFrame(
            probs.reshape(num_beats, elements_per_time_step * 2, order="C").T
        )

        play_bias = 0
        probs = probs + play_bias
        probs[probs > 1] = 1
        probs[probs < 0] = 0
        out = np.zeros_like(probs)
        # out[tst_output_reshaped > 0.5 ] = 1

        # This part fails when we have multiple output.
        out = np.random.binomial(1, probs, size=None).reshape(num_notes * 2)

        # Need to window across out to get the input form again.
        out = out.reshape((256, 1))
        probs = probs.values.reshape((256,))

        outputs.append(
            out.reshape(
                256,
            )
        )
        all_probs.append(probs)

        # this_vicin = total_vicinity-4-12-12-1
        this_vicin = 24
        next_pred, _, _ = MidiSupport().windowed_data_across_notes_time(
            out, mask_length_x=this_vicin, return_labels=False
        )  # Return (total_vicinity, 128)

        # Get array of Midi values for each note value
        n_notes = 128
        midi_row = MidiSupport().add_midi_value(next_pred, n_notes)

        # Get array of one hot encoded pitch values for each note value
        pitchclass_rows = MidiSupport().calculate_pitchclass(midi_row, next_pred)

        # Add total_pitch count repeated for each note window
        previous_context = MidiSupport().build_context(
            next_pred, midi_row, pitchclass_rows
        )

        midi_row = midi_row.reshape((1, -1))
        next_pred = np.vstack((next_pred, midi_row, pitchclass_rows, previous_context))

        last_beats_int += 1
        last_beats_int = last_beats_int % 16
        next_beats_ind = np.array([int(x) for x in bin(last_beats_int)[2:].zfill(4)])
        next_beats_ind = next_beats_ind.reshape((4, 1))
        next_beats_ind = np.repeat(next_beats_ind, num_notes, axis=1)

        last_new_note = np.concatenate([next_pred, next_beats_ind])
        last_new_note = last_new_note[np.newaxis, :, :]  # Shape now  (1, 28, 128)
        last_new_note = np.swapaxes(last_new_note, 1, 2)  # Shape now  (1, 128, 28)
        input_notes_reshape = last_new_note

    outputs_joined = pd.DataFrame(outputs)
    all_probs_joined = pd.DataFrame(all_probs)
    return outputs_joined, all_probs_joined


def predict_notes_note_invariant_plus_extras_multiple_time_steps(
    model, reshaped_train_data, num_beats=15, size=10, note_vicinity=24
):
    """Predict notes
    Same as before but uses sequence_length number of beats to predict output
    """
    outputs = []
    all_probs = []
    num_notes = 128
    total_vicinity = 53
    offset = np.random.choice(range(len(reshaped_train_data)))
    input_notes = reshaped_train_data[offset : offset + num_beats, :, :]
    input_notes_reshape = input_notes
    last_beats = [str(x) for x in input_notes_reshape[0, -1, -4:]]
    last_beats_int = int("".join(last_beats), 2)

    for l in range(size):

        probs = model.predict(input_notes_reshape)
        # probs shape should be something like (256, beats)
        probs = probs.reshape(num_beats, num_notes * 2, order="C").T

        probs = probs[:, -1:]
        # output sequence will be the same length as the input. Try to either take the first or the last beat

        play_bias = 0
        probs = probs + play_bias
        probs[probs > 1] = 1
        probs[probs < 0] = 0
        out = np.zeros_like(probs)

        out = np.random.binomial(1, probs, size=None).reshape(num_notes * 2)

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
        next_pred, _, _ = MidiSupport().windowed_data_across_notes_time(
            out, mask_length_x=note_vicinity, return_labels=False
        )  # Return (total_vicinity, 128)

        # Get array of Midi values for each note value
        n_notes = 128
        midi_row = MidiSupport().add_midi_value(next_pred, n_notes)

        # Get array of one hot encoded pitch values for each note value
        pitchclass_rows = MidiSupport().calculate_pitchclass(midi_row, next_pred)

        # Add total_pitch count repeated for each note window
        previous_context = MidiSupport().build_context(
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

        together = np.concatenate(
            [input_notes_reshape[1:, :, :], last_new_note], axis=0
        )
        input_notes_reshape = together

    outputs_joined = pd.DataFrame(outputs)
    all_probs_joined = pd.DataFrame(all_probs)
    return outputs_joined, all_probs_joined
