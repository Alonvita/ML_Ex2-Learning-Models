import numpy as np
import random
from TrainingSetManipulations import TrainingSetManipulations as TSM

if __name__ == "__main__":
    df = TSM.train_fp_to_data_frame("train_x.txt", "train_y.txt", True)
    print(df)
