import pandas as pd


class TrainingSetManipulations:
    @staticmethod
    def train_fp_to_data_frame(training_fp):
        # read to data frame
        return pd.read_csv(training_fp,
                           delimiter=',',
                           names=["Sex", "Length", "Diameter", "Height",
                                  "Whole Weight", "Shucked Weight",
                                  "Viscera Weight", "Shell Weight"])

    @staticmethod
    def shuffle_data_frame(df):
        return df.sample(frac=1).reset_index(drop=True)


