#!/bin/bash

# Get parameters or use defaults
train=${1:-0.8}
val=${2:-0.1}
test=${3:-0.1}

# Check if parameters are numbers (including floating point)
if ! [[ "$train" =~ ^[0-9]*\.?[0-9]+$ ]] || ! [[ "$val" =~ ^[0-9]*\.?[0-9]+$ ]] || ! [[ "$test" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    echo "Error: Parameters must be numbers (integers or decimals)"
    exit 1
fi

# Calculate sum using bc for floating point arithmetic
sum=$(echo "$train + $val + $test" | bc)

# Check if sum equals 1 using bc
if (( $(echo "$sum != 1" | bc -l) )); then
    echo "Error: The sum of parameters must be 1"
    echo "Current sum is: $sum (train=$train, val=$val, test=$test)"
    exit 1
fi

# If validation passes, call split.py
echo "Running split..."
python split.py --train $train --val $val --test $test
