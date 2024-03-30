#!/bin/bash

input_file="$1"  # CORAA metadata CSV file
output_file="$2" # output CSV file
txt_file="${output_file%.*}.txt" # TXT file of utterances and their X-SAMPA phonetic transcriptions
tmp_file="${output_file%.*}_tmp" # auxiliary file

tail -n +2 "$input_file" | awk -F, '{print $15, "\r"}' | sed -e 's/ /\n/g' | docker run --rm -i falabrasil/g2p 2>/dev/null | awk -F'\t' '{print $2}' | sed 's/  */ /g' > "$tmp_file" # This command runs the g2p and outputs one word per line. The phrases of the input data are separated with double "\n". Breakdown of the command:

# 1: skips header of the CSV file
# 2: outputs the second column ended with "\r"
# 3: replaces spaces with "\n" (falabrasil/g2p does not support spaces)
# 4: run falabrasil/g2p
# 5: outputs the second column of the g2p output
# 6: replaces multiple spaces with a single space

awk '{ORS = (/^$/ ? "\n" : "  ")} 1' "$tmp_file" > "$txt_file" && sed -i 's/[[:space:]]*$//' "$txt_file" # This command does 3 kinds of substitutions:

# 1: awk replaces empty lines with "\n" and
# 2: appends a double space to non-empty lines
# 3: sed trims trailing whitespaces

echo "g2p" | cat - "$txt_file" > "$tmp_file" && paste -d, "$input_file" "$tmp_file" > "$output_file" # concat CSV and TXT files
rm -f "$tmp_file"
