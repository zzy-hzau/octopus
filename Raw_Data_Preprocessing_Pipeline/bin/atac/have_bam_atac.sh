#!/bin/bash

# --- Check Arguments ---
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <INPUT_FILE> <REF_FASTA> <THREADS> <WORK_DIR>"
    echo "Example: $0 ./data/sample.sra ./ref/genome.fa 8 /home/user/project_results"
    exit 1
fi

# Arguments
INPUT=$1      # Path to .sra, _1.fastq.gz, or .fastq
REF_FASTA=$2  # Path to reference genome fasta
THREADS=$3    # Number of CPU cores
WORK_DIR=$4   # The project root directory (e.g., /home/user/project)

# Setup paths based on your requested structure
BASENAME=$(basename "$INPUT" | sed -E 's/\.(sra|fastq\.gz|fastq|fq\.gz|fq)$//; s/_[12]$//')
REF_DIR=$(dirname "$REF_FASTA")
REF_NAME=$(basename "$REF_FASTA")

# Specific sample folder: ${WORK_DIR}/atac_data/${BASENAME}
SAMPLE_DIR="${WORK_DIR}/atac_data/${BASENAME}"
# Global reference files: ${WORK_DIR}/chrom.sizes
CHROM_SIZES="${WORK_DIR}/chrom.sizes"

# Create directories
mkdir -p "$SAMPLE_DIR"

echo "------------------------------------------"
echo "Project Root   : ${WORK_DIR}"
echo "Sample Folder  : ${SAMPLE_DIR}"
echo "Chrom Sizes    : ${CHROM_SIZES}"
echo "------------------------------------------"

# Load environment modules
module load Bowtie2/2.5.2 SAMtools/1.9 fastp sratoolkit/3.0.7 ucsc_kentUtils/v389 deepTools/3.5.0 sambamba

# --- 1. Reference Genome Indexing & Chrom Sizes ---
if [ ! -f "${REF_DIR}/${REF_NAME}.1.bt2" ]; then
    echo "Status: Indexing reference genome..."
    bowtie2-build "$REF_FASTA" "${REF_DIR}/${REF_NAME}"
fi

# Generate chrom.sizes at the root of WORK_DIR
if [ ! -f "$CHROM_SIZES" ]; then
    echo "Status: Generating chrom.sizes..."
    samtools faidx "$REF_FASTA"
    cut -f1,2 "${REF_FASTA}.fai" > "$CHROM_SIZES"
fi

# --- 2. Input Handling ---
if [[ "$INPUT" == *.sra ]]; then
    if [ ! -f "${SAMPLE_DIR}/${BASENAME}_1.fastq" ] && [ ! -f "${SAMPLE_DIR}/${BASENAME}.fastq" ]; then
        echo "Status: Converting SRA to Fastq..."
        fasterq-dump "$INPUT" -O "$SAMPLE_DIR" --split-files --threads "$THREADS"
    fi
elif [[ "$INPUT" == *.fastq* ]] || [[ "$INPUT" == *.fq* ]]; then
    echo "Status: Copying input Fastq files..."
    DIR_NAME=$(dirname "$INPUT")
    if [[ "$INPUT" == *_1.fastq* ]] || [[ "$INPUT" == *_1.fq* ]]; then
        cp ${DIR_NAME}/${BASENAME}_[12].f*q* "$SAMPLE_DIR/"
    else
        cp "$INPUT" "$SAMPLE_DIR/"
    fi
fi

# --- 3. Identify Sequencing Mode ---
if [ -f "${SAMPLE_DIR}/${BASENAME}_1.fastq.gz" ] || [ -f "${SAMPLE_DIR}/${BASENAME}_1.fastq" ] || [ -f "${SAMPLE_DIR}/${BASENAME}_1.fq.gz" ]; then
    MODE="PE"
    IN1=$(ls ${SAMPLE_DIR}/${BASENAME}_1.f*q* | head -n 1)
    IN2=$(ls ${SAMPLE_DIR}/${BASENAME}_2.f*q* | head -n 1)
    OUT1="${SAMPLE_DIR}/${BASENAME}_1.clean.fq.gz"
    OUT2="${SAMPLE_DIR}/${BASENAME}_2.clean.fq.gz"
else
    MODE="SE"
    IN1=$(ls ${SAMPLE_DIR}/${BASENAME}*.f*q* | head -n 1)
    OUT1="${SAMPLE_DIR}/${BASENAME}.clean.fq.gz"
fi

# --- 4. Quality Control & Trimming (fastp) ---
if [ ! -f "$OUT1" ]; then
    echo "Status: Running fastp QC..."
    if [ "$MODE" == "PE" ]; then
        fastp -w "$THREADS" -i "$IN1" -I "$IN2" -o "$OUT1" -O "$OUT2"
        [ -f "$OUT1" ] && rm "$IN1" "$IN2"
    else
        fastp -w "$THREADS" -i "$IN1" -o "$OUT1"
        [ -f "$OUT1" ] && rm "$IN1"
    fi
fi

# --- 5. Alignment (Bowtie2) ---
SORTED_BAM="${SAMPLE_DIR}/${BASENAME}_sorted.bam"
if [ ! -f "$SORTED_BAM" ] && [ ! -f "${SAMPLE_DIR}/${BASENAME}_sorted_rmd.bam" ]; then
    echo "Status: Aligning reads..."
    if [ "$MODE" == "PE" ]; then
        bowtie2 --local --minins 25 --maxins 2000 --no-mixed --no-discordant --dovetail \
            -p "$THREADS" -x "${REF_DIR}/${REF_NAME}" -1 "$OUT1" -2 "$OUT2" | \
            samtools view -@ "$THREADS" -q 30 -bS | \
            samtools sort -@ "$THREADS" -o "$SORTED_BAM"
        [ -f "$SORTED_BAM" ] && rm "$OUT1" "$OUT2"
    else
        bowtie2 --local -p "$THREADS" -x "${REF_DIR}/${REF_NAME}" -U "$OUT1" | \
            samtools view -@ "$THREADS" -q 30 -bS | \
            samtools sort -@ "$THREADS" -o "$SORTED_BAM"
        [ -f "$SORTED_BAM" ] && rm "$OUT1"
    fi
fi

# --- 6. Filtering & Duplicate Removal (sambamba) ---
RMD_BAM="${SAMPLE_DIR}/${BASENAME}_sorted_rmd.bam"
if [ ! -f "$RMD_BAM" ]; then
    echo "Status: Filtering and Deduplicating..."
    sambamba view -t "$THREADS" -f bam -F "mapping_quality >= 30 and ref_name != 'ChrM' and ref_name != 'ChrC' and ref_name != 'MT' and ref_name != 'Pt'" \
        "$SORTED_BAM" > "${SAMPLE_DIR}/filtered.tmp.bam"
    
    sambamba markdup -t "$THREADS" -p --overflow-list-size 600000 --tmpdir='./tmp' -r \
        "${SAMPLE_DIR}/filtered.tmp.bam" "$RMD_BAM"
    
    rm "${SAMPLE_DIR}/filtered.tmp.bam"
    
    # Final log entry in the root project directory
    echo "$(date "+%Y-%m-%d %H:%M:%S") | ${BASENAME} | Done" >> "${WORK_DIR}/pipeline.log"
fi