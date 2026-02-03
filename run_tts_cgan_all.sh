#!/bin/bash

# 🚀 NILM TTS-CGAN All-in-One Automated Pipeline
# This script trains each appliance one-by-one and generates 200% synthetic data.

APPLIANCES=("dishwasher" "fridge" "kettle" "microwave" "washingmachine")
EPOCHS=100
BATCH_SIZE=256
SEQ_LEN=512

for APP in "${APPLIANCES[@]}"
do
    echo "========================================================="
    echo "STARTING PROCESS FOR: $APP"
    echo "========================================================="

    # 1. Train the model
    echo "Step 1: Training TTS-CGAN for $APP..."
    python NILM_Train_TTS_CGAN.py --appliance $APP --epochs $EPOCHS --batch_size $BATCH_SIZE

    # 2. Calculate required samples (200% of original)
    # Count lines in original CSV, divide by seq_len, then multiply by 2
    CSV_FILE="./data/${APP}_training_.csv"
    if [ -f "$CSV_FILE" ]; then
        LINE_COUNT=$(wc -l < "$CSV_FILE")
        # Subtract 1 for header
        ROW_COUNT=$((LINE_COUNT - 1))
        # Original windows = rows / 512
        ORIG_WINDOWS=$((ROW_COUNT / SEQ_LEN))
        # 200% Windows = Original * 2
        TARGET_SAMPLES=$((ORIG_WINDOWS * 2))
        
        echo "Original Rows: $ROW_COUNT -> Original Windows: $ORIG_WINDOWS"
        echo "Step 2: Generating $TARGET_SAMPLES synthetic windows (200%)..."

        # 3. Generate data
        python generate_synthetic_tts_cgan.py --appliance $APP --num_samples $TARGET_SAMPLES
    else
        echo "Error: Data file $CSV_FILE not found. Skipping generation."
    fi

    echo "COMPLETED: $APP"
    echo ""
done

echo "ALL APPLIANCES DONE. Check 'synthetic_out/' for results."
