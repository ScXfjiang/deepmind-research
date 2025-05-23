import tensorflow as tf
import json, pathlib, collections

# enable eager mode
tf.compat.v1.enable_eager_execution()

tfrecord_path = "train.tfrecord"
meta_path = "meta.json"


def describe_example(serialised):
    """Return {key: (dtype, flat_length)} for one tf.train.Example."""
    example = tf.train.Example.FromString(serialised.numpy())
    out = {}
    for k, feature in example.features.feature.items():
        kind = feature.WhichOneof("kind")
        if kind == "bytes_list":
            out[k] = ("bytes", len(feature.bytes_list.value))
        elif kind == "float_list":
            out[k] = ("float32", len(feature.float_list.value))
        else:
            out[k] = ("int64", len(feature.int64_list.value))
    return out


# 1) scan the first example
summary = collections.defaultdict(set)
raw_ds = tf.data.TFRecordDataset(tfrecord_path)
for raw in raw_ds.take(1):
    for k, (dtype, length) in describe_example(raw).items():
        summary[k].add((dtype, length))

# 2) print the feature summary of the first example
print("\n=== TFRecord feature summary ===")
for k, specs in summary.items():
    dt = {d for d, _ in specs}
    lens = {l for _, l in specs}
    print(f"{k:<20}: dtype={','.join(dt):7} flat_length={lens}")

# 3) cross-reference with meta.json
if meta_path and pathlib.Path(meta_path).exists():
    meta = json.load(open(meta_path))
    print("\n=== Shapes from meta.json ===")
    for k, info in meta["features"].items():
        shape = info["shape"]
        print(f"{k:<20}: {shape}, dtype={info['dtype']}  ({info['type']})")
