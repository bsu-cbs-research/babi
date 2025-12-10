# python -m data.scripts.preprocess
from data.parse import capnostream, masimo, mocap

if __name__ == "__main__":
    capnostream.parse_preprocessed_locals()
    mocap.parse()
    mocap.parse(with_tuples=True)
    masimo.parse()