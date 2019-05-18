import TrainingSetManipulations
from TrainingSetManipulations import TrainingSetManipulations as TSM

if __name__ == "__main__":
    df = TSM.train_fp_to_data_frame("train_x.txt", "train_y.txt", True)
    for i in df[TrainingSetManipulations.DATA_SET_X_VALUES].values:
        print(i)
