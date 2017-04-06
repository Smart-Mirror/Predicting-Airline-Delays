import pydoop.hdfs as hdfs


def read_csv_from_hdfs(path, cols, col_types=None):
    files = hdfs.ls(path)
    pieces = []
    for f in files:
        handle = hdfs.open(f)
        pieces.append(pd.read_csv(handle, names=cols, dtype=col_types))
        handle.close()
    return pd.concat(pieces, ignore_index=True)
