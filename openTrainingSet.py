import pandas as pd


def train_fp_to_data_frame(training_fp):
    # read to data frame
    return pd.read_csv(training_fp, delimiter=',',
                       names=["Sex", "Length", "Diameter", "Height",
                              "Whole Weight", "Shucked Weight",
                              "Viscera Weight", "Shell Weight"])


if __name__ == "__main__":
    df = train_fp_to_data_frame("train_x.txt")

    print(df)




