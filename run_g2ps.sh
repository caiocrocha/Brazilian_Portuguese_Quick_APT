#!/bin/bash

# Input and output file paths
input_file="$1"
output_dir="$2"
output_prefix="$3"

if [ $# -lt 3 ]; then
	echo "Missing arguments"
	echo "1: input text file"
	echo "2: output directory"
	echo "3: prefix of output files, which will be saved as OUTPUT_DIR/PREFIX_G2P.csv"
	exit
fi

mkdir -p "$output_dir"

echo "Falabrasil G2P"
./falabrasil_g2p.sh "$input_file" "${output_dir}/${output_prefix}_falabrasil.csv" || echo "Error running falabrasil G2P"

echo "espeak G2P"
./espeak_g2p.sh "$input_file" "${output_dir}/${output_prefix}_espeak.csv" || echo "Error running espeak G2P"

echo "espeak-ng G2P"
./espeakng_g2p.sh "$input_file" "${output_dir}/${output_prefix}_espeak-ng.csv" || echo "Error running espeak-ng G2P"

echo "Phonetisaurus G2P"
./phonetisaurus_g2p.sh "$input_file" "${output_dir}/${output_prefix}_phonetisaurus.csv" || echo "Error running phonetisaurus G2P"
