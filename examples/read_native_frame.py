"""Read a frame through the installed native API without producing another PHDF."""

import argparse
import json

from dense_dump_codec.native import NativeSequenceDecoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sequence", type=int, default=13)
    parser.add_argument("--working-cache")
    arguments = parser.parse_args()
    decoder = NativeSequenceDecoder(arguments.manifest, workers=2,
                                    working_cache=arguments.working_cache)
    frame = decoder([arguments.sequence])[arguments.sequence]
    print(json.dumps({"sequence": frame["sequence"], "time": frame["time"],
                      "arrays": {name: {"shape": list(array.shape), "dtype": str(array.dtype)}
                                 for name, array in frame["datasets"].items()},
                      "access": decoder.access_statistics()}, indent=2))
