from src.experiments import RNNMusicExperimentEleven

# Main training loop
if __name__ == "__main__":
    
    learning_rate = 0.001
    training_epoch = 1
    
    print("Trying Exp 11")
    exp = RNNMusicExperimentEleven(learning_rate=0.0001, num_music_files=1, dropout=0)
    exp.run()
