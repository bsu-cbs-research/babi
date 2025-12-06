# python -m data.scripts.preprocess
from data.preprocess import capnostream, masimo, mocap

if __name__ == "__main__":
    capnostream.preprocess()
    mocap.preprocess()
    mocap.preprocess(with_tuples=True)
    masimo.preprocess()