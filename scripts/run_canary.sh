#!/bin/bash
# Canary Launch Helper
# Usage: ./scripts/run_canary.sh [TESTNET|LIVE]

MODE=${1:-TESTNET}

echo "=== CANARY LAUNCHER: $MODE ==="
export LIVE_TRADING="YES_I_UNDERSTAND"

# 1. Get Hash (Fail intentionally)
echo "Fetching Config Hash..."
OUTPUT=$(PYTHONPATH=. python3 live/run_paper.py --mode $MODE 2>&1)
HASH=$(echo "$OUTPUT" | grep "RealHash=" | awk -F'RealHash=' '{print $2}')

if [ -z "$HASH" ]; then
    echo "Could not extract hash. Output:"
    echo "$OUTPUT"
    exit 1
fi

echo "Detected Hash: $HASH"
echo "Arming System..."

# 2. Run with Confirm
PYTHONPATH=. python3 live/run_paper.py --mode $MODE --live_confirm $HASH
