#!/bin/bash

# Input and output file paths
input_file="$1"
output_file="$2"

cat "$input_file" | docker run --rm -i falabrasil/g2p | awk '{print $1 "," substr($0, length($1) + 2)}' > "${output_file}"
