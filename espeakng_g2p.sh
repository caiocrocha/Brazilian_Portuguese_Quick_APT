#!/bin/bash

# Input and output file paths
input_file="$1"
output_file="$2"

espeak-ng -f $input_file -v pt-br -q -x --ipa | sed -e 's/[ːˈˌ]//g' -e 's/  */\n/g' | grep "\S" > "${output_file%.*}.txt" && paste -d, $input_file "${output_file%.*}.txt" > "${output_file}"
