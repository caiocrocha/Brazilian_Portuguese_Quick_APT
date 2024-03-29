#!/bin/bash

# Input and output file paths
input_file="$1"
output_file="$2"

# Model path
phonetisaurus_model_path="/home/caio/Documents/github/g2ps/models/portuguese.fst"

phonetisaurus-g2pfst --model="$phonetisaurus_model_path" --wordlist="$input_file" | awk '{print $1 "," substr($0, length($1 $2) + 3)}' > "${output_file}"
