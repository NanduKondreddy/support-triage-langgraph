#!/bin/bash
set -e
echo "Checking dependencies..."
# This project has zero external dependencies!
# (pip install -r requirements.txt is a no-op)

echo "Running regression tests..."
python code/tests/test_regression.py

echo "Running on full input..."
python code/main.py --input support_tickets/support_tickets.csv --output support_tickets/output.csv

echo "Output ready at support_tickets/output.csv"
