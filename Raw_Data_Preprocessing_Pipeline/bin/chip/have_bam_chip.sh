#!/bin/bash

# --- Argument Check ---
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <INPUT_FILE> <REF_FASTA> <THREADS> <WORK_DIR>"
    echo "Example: $0 ./data/sample.sra ./ref/genome.fa 8 /home/user/chip_project"
    exit 1
fi

# Arguments
INPUT=$1      # Path to .sra, _1.fastq.gz, or .fastq
REF_FASTA=$2  # Path to reference genome fasta
THREADS=$3    # Number of CPU cores
WORK_DIR=$4   # Mandatory project root directory

# Setup paths and naming
# Extract sample ID and define directories
BASENAME=$(basename "$INPUT" | sed -E 's/\.(sra|fastq\.gz|fastq|fq\.gz|fq)$//; s/_[12]$//')
REF_DIR=$(dirname "$REF_FASTA")
REF_NAME=$(basename "$REF_FASTA")

# Defined structure: ${WORK_DIR}/chip_seq/${BASENAME}
SAMPLE_DIR="${WORK_DIR}/chip_seq/${BASENAME}"
CHROM_SIZES="${WORK_DIR}/chrom.sizes"

mkdir -p "$SAMPLE_DIR"
echo "------------------------------------------"
echo "Processing ChIP-seq Sample : ${BASENAME}"
echo "Output Directory           : ${SAMPLE_DIR}"
echo "Threads                    : ${THREADS}"
echo "------------------------------------------"

# Load environment modules
module load Bowtie2/2.5.2 SAMtools/1.9 fastp sratoolkit/3.0.7 ucsc_kentUtils/v389 deepTools/3.5.0 sambamba

# --- 1. Reference Indexing & Chrom Sizes ---
if [ ! -f "${REF_DIR}/${REF_NAME}.1.bt2" ]; then
    echo "Status: Indexing reference genome..."
    bowtie2-build "$REF_FASTA" "${REF_DIR}/${REF_NAME}"
fi

if [ ! -f "$CHROM_SIZES" ]; then
    echo "Status: Generating chrom.sizes..."
    samtools faidx "$REF_FASTA"
    cut -f1,2 "${REF_FASTA}.fai" > "$CHROM_SIZES"
fi

# --- 2. Input Handling (SRA/Fastq) ---
# Check and copy or convert to workspace
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

# --- 3. Identify Mode & Native Gzip Handling ---
# No manual unzip; tools read .gz directly
if [ -f "${SAMPLE_DIR}/${BASENAME}_1.fastq.gz" ] || [ -f "${SAMPLE_DIR}/${BASENAME}_1.fastq" ]; then
    MODE="PE"
    IN1=$(ls ${SAMPLE_DIR}/${BASENAME}_1.f*q* | head -n 1)
    IN2=$(ls ${SAMPLE_DIR}/${BASENAME}_2.f*q* | head -n 1)
    OUT1="${SAMPLE_DIR}/${BASENAME}_1.clean.fq.gz"
    OUT2="${SAMPLE_DIR}/${BASENAME}_2.clean.fq.gz"
    echo "Detected Mode: Paired-End"
else
    MODE="SE"
    # Find either basename.fastq or basename_1.fastq as single end
    IN1=$(ls ${SAMPLE_DIR}/${BASENAME}*.f*q* | head -n 1)
    OUT1="${SAMPLE_DIR}/${BASENAME}.clean.fq.gz"
    echo "Detected Mode: Single-End"
fi

# --- 4. Quality Control & Trimming (fastp) ---
if [ ! -f "$OUT1" ]; then
    echo "Status: Running fastp QC..."
    if [ "$MODE" == "PE" ]; then
        fastp -w "$THREADS" -i "$IN1" -I "$IN2" -o "$OUT1" -O "$OUT2" \
              -h "${SAMPLE_DIR}/${BASENAME}_fastp.html" -j "${SAMPLE_DIR}/${BASENAME}_fastp.json"
        [ -f "$OUT1" ] && rm "$IN1" "$IN2"
    else
        fastp -w "$THREADS" -i "$IN1" -o "$OUT1" \
              -h "${SAMPLE_DIR}/${BASENAME}_fastp.html" -j "${SAMPLE_DIR}/${BASENAME}_fastp.json"
        [ -f "$OUT1" ] && rm "$IN1"
    fi
fi

# --- 5. Alignment (Bowtie2) ---
SORTED_BAM="${SAMPLE_DIR}/${BASENAME}_sorted.bam"
if [ ! -f "$SORTED_BAM" ] && [ ! -f "${SAMPLE_DIR}/${BASENAME}_sorted_rmd.bam" ]; then
    echo "Status: Aligning with Bowtie2..."
    if [ "$MODE" == "PE" ]; then
        # ChIP-seq parameters: --minins 25 --maxins 1500
        bowtie2 --local --minins 25 --maxins 1500 --dovetail --soft-clipped-unmapped-tlen \
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
    echo "Status: Removing Duplicates and Organelle DNA..."
    # Filter for quality 30 and remove Mitochondrial/Chloroplast reads
    sambamba view -t "$THREADS" -f bam -F "mapping_quality >= 30 and ref_name != 'ChrM' and ref_name != 'ChrC' and ref_name != 'MT' and ref_name != 'Pt'" \
        "$SORTED_BAM" > "${SAMPLE_DIR}/filtered.tmp.bam"
    
    sambamba markdup -t "$THREADS" -p --overflow-list-size 600000 --tmpdir='./tmp' -r \
        "${SAMPLE_DIR}/filtered.tmp.bam" "$RMD_BAM"
    
    # Clean up temp files
    rm "${SAMPLE_DIR}/filtered.tmp.bam"
    # rm "$SORTED_BAM" # Keep or remove sorted.bam based on storage policy
    
    # Log progress to the root project directory
    echo "$(date "+%Y-%m-%d %H:%M:%S") | ChIP-seq | ${BASENAME} | Done" >> "${WORK_DIR}/pipeline.log"
fi