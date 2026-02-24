#!/bin/bash

# --- Argument Check ---
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <INPUT_BAM> <WORK_DIR> <BASENAME> <THREADS>"
    echo "Example: $0 ./results/chip_seq/S1/S1_sorted_rmd.bam /home/user/chip_project S1 8"
    exit 1
fi

# Arguments
INPUT_BAM=$1   # Path to the input BAM file (rmd.bam)
WORK_DIR=$2    # The project root directory
BASENAME=$3    # Sample name for the output file
THREADS=$4     # Number of CPU cores

# Setup paths based on standard structure
OUT_DIR="${WORK_DIR}/bw"
OUTPUT_BW="${OUT_DIR}/${BASENAME}.bw"
LOG_FILE="${WORK_DIR}/pipeline.log"

# Load environment modules
module load deepTools/3.5.0
module load SAMtools/1.9

# --- 1. Directory Preparation ---
if [ ! -d "$OUT_DIR" ]; then
    echo "Status: Creating BigWig directory at $OUT_DIR..."
    mkdir -p "$OUT_DIR"
fi

# --- 2. Indexing ---
# bamCoverage requires an indexed BAM file (.bai)
if [ ! -f "${INPUT_BAM}.bai" ]; then
    echo "Status: Indexing BAM file..."
    samtools index -@ "$THREADS" "$INPUT_BAM"
fi

# --- 3. Generate BigWig ---
# Normalize using RPKM with 1bp binSize for high-resolution visualization
echo "------------------------------------------"
echo "Input BAM   : ${INPUT_BAM}"
echo "Output BW   : ${OUTPUT_BW}"
echo "Normalization: RPKM"
echo "------------------------------------------"

bamCoverage \
    --bam "$INPUT_BAM" \
    --outFileName "$OUTPUT_BW" \
    --outFileFormat bigwig \
    --binSize 1 \
    --numberOfProcessors "$THREADS" \
    --normalizeUsing RPKM \
    --ignoreDuplicates

# --- 4. Logging ---
if [ $? -eq 0 ]; then
    CURRENT_TIME=$(date "+%Y-%m-%d %H:%M:%S")
    # Log to the centralized project log in WORK_DIR
    echo "${CURRENT_TIME} | ChIP-seq | ${BASENAME} | BW Success" >> "$LOG_FILE"
    echo "Processing complete: ${OUTPUT_BW}"
else
    echo "Error: bamCoverage failed for ${BASENAME}"
    echo "$(date "+%Y-%m-%d %H:%M:%S") | ChIP-seq | ${BASENAME} | BW FAILED" >> "$LOG_FILE"
    exit 1
fi