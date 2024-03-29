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
cat "$input_file" | docker run --rm -i falabrasil/g2p | awk '{print $1 "," substr($0, length($1) + 2)}' > "${output_dir}/${output_prefix}_falabrasil.csv"

echo "espeak G2P"
espeak -f $input_file -v pt -q -x --ipa | sed -e 's/[ːˈˌ]//g' -e 's/  */\n/g' | grep "\S" > "${output_dir}/${output_prefix}_espeak.txt" && paste -d, $input_file "${output_dir}/${output_prefix}_espeak.txt" > "${output_dir}/${output_prefix}_espeak.csv"

echo "espeak-ng G2P"
espeak-ng -f $input_file -v pt-br -q -x --ipa | sed -e 's/[ːˈˌ]//g' -e 's/  */\n/g' | grep "\S" > "${output_dir}/${output_prefix}_espeak-ng.txt" && paste -d, $input_file "${output_dir}/${output_prefix}_espeak-ng.txt" > "${output_dir}/${output_prefix}_espeak-ng.csv"

echo "Phonetisaurus G2P"
phonetisaurus_model_path="/home/caio/Documents/github/g2ps/models/portuguese.fst"
phonetisaurus-g2pfst --model="$phonetisaurus_model_path" --wordlist="$input_file" | awk '{print $1 "," substr($0, length($1 $2) + 3)}' > "${output_dir}/${output_prefix}_phonetisaurus.csv"
