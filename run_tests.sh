#!/bin/bash
# UFR Autoresearch Test Scripts
# Usage: ./run_tests.sh [1h|3h|overnight]

set -e

TEST_TYPE=${1:-1h}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Ensure data is prepared
if [ ! -d "$HOME/.cache/autoresearch/data" ]; then
    echo "Data not found. Running prepare.py first..."
    uv run prepare.py
fi

# Create results directory
mkdir -p results

echo "=========================================="
echo "UFR Autoresearch Test Suite"
echo "Test type: $TEST_TYPE"
echo "Start time: $(date)"
echo "=========================================="

case $TEST_TYPE in
    1h)
        echo "Running 1-hour baseline test..."
        echo "This will run ~12 experiments (5 min each)"
        END_TIME=$(($(date +%s) + 3600))
        EXP_COUNT=0
        while [ $(date +%s) -lt $END_TIME ]; do
            EXP_COUNT=$((EXP_COUNT + 1))
            echo ""
            echo "=== Experiment $EXP_COUNT ==="
            uv run train_ufr.py 2>&1 | tee "results/exp_${TEST_TYPE}_${EXP_COUNT}_$(date +%H%M%S).log"
            
            # Save metrics
            if [ -f metrics_output.json ]; then
                cp metrics_output.json "results/metrics_${TEST_TYPE}_${EXP_COUNT}.json"
            fi
            
            # Git commit if improved
            git add -A
            git commit -m "1h test: experiment $EXP_COUNT" || true
        done
        echo "Completed $EXP_COUNT experiments in 1 hour"
        ;;
        
    3h)
        echo "Running 3-hour exploration test..."
        echo "This will run ~36 experiments (5 min each)"
        END_TIME=$(($(date +%s) + 10800))
        EXP_COUNT=0
        while [ $(date +%s) -lt $END_TIME ]; do
            EXP_COUNT=$((EXP_COUNT + 1))
            echo ""
            echo "=== Experiment $EXP_COUNT ==="
            uv run train_ufr.py 2>&1 | tee "results/exp_${TEST_TYPE}_${EXP_COUNT}_$(date +%H%M%S).log"
            
            if [ -f metrics_output.json ]; then
                cp metrics_output.json "results/metrics_${TEST_TYPE}_${EXP_COUNT}.json"
            fi
            
            git add -A
            git commit -m "3h test: experiment $EXP_COUNT" || true
        done
        echo "Completed $EXP_COUNT experiments in 3 hours"
        ;;
        
    overnight)
        echo "Running overnight test (~8-10 hours)..."
        echo "This will run ~100 experiments"
        END_TIME=$(($(date +%s) + 36000))  # 10 hours
        EXP_COUNT=0
        while [ $(date +%s) -lt $END_TIME ]; do
            EXP_COUNT=$((EXP_COUNT + 1))
            echo ""
            echo "=== Experiment $EXP_COUNT ==="
            uv run train_ufr.py 2>&1 | tee "results/exp_${TEST_TYPE}_${EXP_COUNT}_$(date +%H%M%S).log"
            
            if [ -f metrics_output.json ]; then
                cp metrics_output.json "results/metrics_${TEST_TYPE}_${EXP_COUNT}.json"
            fi
            
            git add -A
            git commit -m "Overnight test: experiment $EXP_COUNT" || true
            
            # Progress report every 10 experiments
            if [ $((EXP_COUNT % 10)) -eq 0 ]; then
                echo "Progress: $EXP_COUNT experiments completed at $(date)"
            fi
        done
        echo "Completed $EXP_COUNT experiments overnight"
        ;;
        
    *)
        echo "Unknown test type: $TEST_TYPE"
        echo "Usage: ./run_tests.sh [1h|3h|overnight]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Test complete!"
echo "Results saved in: results/"
echo "End time: $(date)"
echo "=========================================="

# Generate summary
if command -v python3 &> /dev/null; then
    python3 << 'PYEOF'
import json
import glob
import os

print("\n=== EXPERIMENT SUMMARY ===")
metrics_files = glob.glob("results/metrics_*.json")
if not metrics_files:
    print("No metrics files found")
else:
    results = []
    for f in sorted(metrics_files):
        try:
            with open(f) as fp:
                data = json.load(fp)
                results.append({
                    'file': os.path.basename(f),
                    'val_bpb': data.get('val_bpb', 0),
                    'composite_score': data.get('structural', {}).get('composite_score', 0),
                    'depth': data.get('model', {}).get('depth', 0),
                })
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    if results:
        print(f"\nTotal experiments: {len(results)}")
        print(f"Best val_bpb: {min(r['val_bpb'] for r in results):.6f}")
        print(f"Best composite_score: {max(r['composite_score'] for r in results):.6f}")
        print("\nTop 5 by composite_score:")
        for r in sorted(results, key=lambda x: x['composite_score'], reverse=True)[:5]:
            print(f"  {r['file']}: bpb={r['val_bpb']:.4f}, score={r['composite_score']:.4f}")
PYEOF
fi
