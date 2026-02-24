#!/bin/bash
# If there are two biological replicates
# --- Check Arguments ---
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <REP1_INPUT> <REP2_INPUT> <REF_FASTA> <PART_NAME> <WORK_DIR>"
    echo "Example: $0 ./data/R1.sra ./data/R2.sra ./ref/gen.fa MySample /home/user/project"
    exit 1
fi

# Arguments
REP1_IN=$1      # Input for Replicate 1
REP2_IN=$2      # Input for Replicate 2
REF=$3          # Reference Fasta
PART=$4         # Merged sample name (e.g., 'WT_Combined')
WORK_DIR=$5     # Project root directory
THREADS=8      # You can also set this as an argument if preferred

# Extract IDs for sub-folder navigation
ID1=$(basename "$REP1_IN" | sed -E 's/\.(sra|fastq\.gz|fastq|fq\.gz|fq)$//; s/_[12]$//')
ID2=$(basename "$REP2_IN" | sed -E 's/\.(sra|fastq\.gz|fastq|fq\.gz|fq)$//; s/_[12]$//')

# Define Paths
# Replicate BAMs are in: ${WORK_DIR}/atac_data/${ID}/
BAM1="${WORK_DIR}/atac_data/${ID1}/${ID1}_sorted_rmd.bam"
BAM2="${WORK_DIR}/atac_data/${ID2}/${ID2}_sorted_rmd.bam"

# Merged outputs
MERGED_DIR="${WORK_DIR}/atac_data/${PART}_merged"
MERGED_BAM="${MERGED_DIR}/${PART}.merged.bam"
LOG_FILE="${WORK_DIR}/pipeline.log"

module load SAMtools/1.9

# --- 1. Process Replicates Individually ---
# Ensure each replicate is processed to generate the _sorted_rmd.bam
echo "Status: Processing individual replicates..."
bash ./bin/atac/have_bam_atac.sh "$REP1_IN" "$REF" "$THREADS" "$WORK_DIR" && \
bash ./bin/atac/have_bam_atac.sh "$REP2_IN" "$REF" "$THREADS" "$WORK_DIR"

# --- 2. Merge Biological Replicates ---
if [ ! -f "$MERGED_BAM" ]; then
    if [ -f "$BAM1" ] && [ -f "$BAM2" ]; then
        echo "Status: Merging replicates into $MERGED_BAM"
        mkdir -p "$MERGED_DIR"
        samtools merge -@ "$THREADS" "$MERGED_BAM" "$BAM1" "$BAM2"
        samtools index -@ "$THREADS" "$MERGED_BAM"
        echo "$(date "+%Y-%m-%d %H:%M:%S") | ${PART} | Merging Success" >> "$LOG_FILE"
    else
        echo "Error: One or both replicate BAMs are missing!"
        echo "$(date "+%Y-%m-%d %H:%M:%S") | ${PART} | Merging FAILED" >> "$LOG_FILE"
        exit 1
    fi
fi

# --- 3. Generate BigWig for the Merged Sample ---
# Path: ${WORK_DIR}/bw/${PART}.bw
echo "Status: Starting BigWig generation..."
bash ./bin/atac/bam_to_bw_atac.sh "$MERGED_BAM" "$WORK_DIR" "$PART" "$THREADS"

echo "------------------------------------------------"
echo "Pipeline Finished for ${PART}"
echo "Merged BAM: ${MERGED_BAM}"
echo "BigWig    : ${WORK_DIR}/bw/${PART}.bw"
echo "------------------------------------------------"

