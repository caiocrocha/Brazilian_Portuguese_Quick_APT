#!/bin/bash

# Arguments:
# 1: Input txt file with the audio IDs
# 2: Output CSV file with the audio IDs and audio durations

while read file
do
    echo -n "$file," && soxi -D "$file"
done < $1 > $2