import pandas as pd

DATA_SET_X_VALUES = ["Sex", "Length", "Diameter", "Height",
                     "Whole Weight", "Shucked Weight",
                     "Viscera Weight", "Shell Weight"]

DATA_SET_Y_VALUES = ["Y_Values"]


class TrainingSetManipulations:
    @staticmethod
    def train_fp_to_data_frame(x_training_fp, y_training_fp, apply_shuffle=False):
        # read to data frame
        x_df = pd.read_csv(x_training_fp,
                           delimiter=',',
                           names=DATA_SET_X_VALUES)

        y_df = pd.read_csv(y_training_fp, names=DATA_SET_Y_VALUES)

        if len(x_df) != len(y_df):
            raise ValueError("Make sures files are of the same size!")

        # join x to y
        joined_df = pd.DataFrame(x_df.join(y_df))

        # check apply_shuffle
        if apply_shuffle:
            joined_df = joined_df.sample(frac=1).reset_index(drop=True)

        return joined_df


