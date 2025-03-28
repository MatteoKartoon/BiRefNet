#!/bin/bash

# Get parameters or use defaults
train=${1:-80}
val=${2:-10}
test=${3:-10}

# Check if parameters are numbers
if ! [[ "$train" =~ ^[0-9]+$ ]] || ! [[ "$val" =~ ^[0-9]+$ ]] || ! [[ "$test" =~ ^[0-9]+$ ]]; then
    echo "Error: Parameters must be numbers"
    exit 1
fi

# Calculate sum
sum=$((train + val + test))

# Check if sum equals 100
if [ $sum -ne 100 ]; then
    echo "Error: The sum of parameters must be 100"
    echo "Current sum is: $sum (train=$train, val=$val, test=$test)"
    exit 1
fi

# If validation passes, call split.py
echo "Running split with ratio $train/$val/$test"
python split.py --train $train --val $val --test $test
