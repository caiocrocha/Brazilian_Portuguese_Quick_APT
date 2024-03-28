#!/bin/bash

# Input and output file paths
input_file="pronunciations/pronunciation_sp_val.txt"
output_file="pronunciations/pronunciation_sp_val"

cat "$input_file" | docker run --rm -i falabrasil/g2p | awk '{print $1 "," substr($0, length($1) + 2)}' > "${output_file}_falabrasil.csv"


> "${output_file}_espeak.csv"

# Loop through each line in the input file
while IFS=, read -r word
do
    # Use espeak to convert the word to phonemes
    phonemes=$(echo "$word" | espeak --stdin -v pt -q -x --ipa=3 | sed -e '/^$/d' -e 's/ː//g' -e 's/ˈ//g' -e 's/ˌ//g' -e 's/\n//g')

    # Append the word ID, word, and phonemes to the output file
    echo "$word,$phonemes" >> "${output_file}_espeak.csv"
done < "$input_file"

phonetisaurus_model_path="/home/caio/Documents/github/g2ps/models/portuguese.fst"

phonetisaurus-g2pfst --model="$phonetisaurus_model_path" --wordlist="$input_file" | awk '{print $1 "," substr($0, length($1 $2) + 3)}' > "${output_file}_phonetisaurus.csv"
