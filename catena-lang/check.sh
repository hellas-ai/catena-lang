#!/usr/bin/env bash

cargo run -- stdlib/cmc.hex stdlib/value.hex stdlib/buf.hex stdlib/index.hex stdlib/base/* stdlib/gpu.hex $1 --output-dir report
