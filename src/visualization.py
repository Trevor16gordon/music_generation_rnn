import matplotlib.pyplot as plt
from src.midi_support import MidiSupport
from IPython import display

def note_and_artic_to_one(data, what="artic"):
    """Converts  an array 256x?? to 128x??.


    In training I'm typically working with play and articulation for all notes.
    Right now I can pull out every second row to have just one.

    Args:
        data (pd.DataFrame): Input data with shape (x, 256)
        what (str, optional): "artic", "note_hold" or "both". Defaults to "artic".

    Raises:
        KeyError: if "what" is incorrect

    Returns:
        np.array:
    """
    with_volume = 80 * data.T.values
    if what == "artic":
        ret = with_volume[range(1, 257, 2), :]
    elif what == "note_hold":
        ret = with_volume[range(0, 256, 2), :]
    elif what == "both":
        ret = with_volume
    else:
        raise KeyError("what needs to be artic or note_hold or both")
    return ret

def plot_piano_roll(note_df, file_path, plot_type="both"):
    """Create and save matplotlib plot of the piano roll.

    Args:
        note_df (): [description]
        file_path ([type]): [description]
        plot_type (str, optional): [description]. Defaults to "both".
    """
    data = note_and_artic_to_one(note_df, what=plot_type)
    plt.rcParams["figure.figsize"] = (40,10)
    plt.imshow(data, cmap='hot', interpolation='nearest', aspect="auto")
    plt.savefig(file_path)

def save_audio_file(predicted, filepath, audio_type="artic", fs=5):
    """Render piano roll DataFrame to audio and save.

    Args:
        predicted (pd.DataFrame): Predicted notes. Should be (x, 256)
        filepath (str): filepath to save
        audio_type (str, optional):  Defaults to "artic".
        fs (int): sampling frequency. The speed of the played audio

    Raises:
        KeyError: [description]
    """
    if audio_type not in ["artic", "note_hold"]:
        raise KeyError("what needs to be artic or note_hold")
    data = note_and_artic_to_one(predicted, what=audio_type)
    new_midi = MidiSupport().piano_roll_to_pretty_midi(data, fs=fs)
    _SAMPLING_RATE = 16000
    seconds = 30
    waveform = new_midi.fluidsynth(fs=_SAMPLING_RATE)
    # Take a sample of the generated waveform to mitigate kernel resets
    waveform_short = waveform[:seconds*_SAMPLING_RATE]
    audtst = display.Audio(waveform_short, rate=_SAMPLING_RATE)
    with open(filepath, "wb") as f:
        f.write(audtst.data)

    new_midi.write(filepath.replace(".wav", ".mid"))
    
