#!/bin/bash

# Get the directory of the script and set base directory to its parent
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$( dirname "$SCRIPT_DIR" )"

# Set up variables
HOST="localhost"
PORT="7670"
OUTPUT_DIR="./processed/smoke_test"
DATA_DIR="${BASE_DIR}/../data"
BATCH_SIZE=32
LOG_LEVEL="WARNING"
EXTRACT_PARAMS='"extract_tables": "True", "extract_charts": "True", "extract_infographics": "True", "extract_images": "True", "extract_text": "True"'

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Create log directory
LOG_DIR="./extraction_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/extraction_test_$(date +%Y%m%d_%H%M%S).log"

# Start log file
echo "====== Multimodal Extraction Test - $(date) ======" | tee -a "$LOG_FILE"
echo "Running from base directory: $BASE_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Function to run extraction and log results
run_extraction() {
    local file_ext=$1
    local doc_type=$2
    local test_file="$DATA_DIR/multimodal_test.$file_ext"

    echo "====== Testing $file_ext extraction ======" | tee -a "$LOG_FILE"
    echo "File: $test_file" | tee -a "$LOG_FILE"
    echo "Document type: $doc_type" | tee -a "$LOG_FILE"
    echo "Started at: $(date)" | tee -a "$LOG_FILE"

    # Check if file exists
    if [ ! -f "$test_file" ]; then
        echo "ERROR: File $test_file not found. Skipping test." | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        return 1
    fi

    # Run extraction
    start_time=$(date +%s)

    nv-ingest-cli --batch_size=$BATCH_SIZE --doc="$test_file" --client_host=$HOST --client_port=$PORT \
      --output_directory="$OUTPUT_DIR" \
      --task="extract:{\"document_type\": \"$doc_type\", $EXTRACT_PARAMS}" --log_level=$LOG_LEVEL \
      2>&1 | tee -a "$LOG_FILE"

    exit_code=${PIPESTATUS[0]}
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo "Exit code: $exit_code" | tee -a "$LOG_FILE"
    echo "Duration: $duration seconds" | tee -a "$LOG_FILE"
    echo "Completed at: $(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # Add separator for readability
    echo "--------------------------------------------------" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    return $exit_code
}

# Counter variables
total_formats=0
successful_formats=0

# Test each file format
declare -A formats=(
    ["pdf"]="pdf"
    ["jpeg"]="jpeg"
    ["png"]="png"
    ["svg"]="svg"
    ["tiff"]="tiff"
    ["bmp"]="bmp"
    ["wav"]="wav"
    ["pptx"]="pptx"
    ["docx"]="docx"
)

# Run tests for each format
for ext in "${!formats[@]}"; do
    doc_type="${formats[$ext]}"
    ((total_formats++))

    run_extraction "$ext" "$doc_type"
    if [ $? -eq 0 ]; then
        ((successful_formats++))
    fi
done

# Print summary
echo "====== SUMMARY ======" | tee -a "$LOG_FILE"
echo "Total formats tested: $total_formats" | tee -a "$LOG_FILE"
echo "Successful extractions: $successful_formats" | tee -a "$LOG_FILE"
echo "Failed extractions: $((total_formats - successful_formats))" | tee -a "$LOG_FILE"
echo "Success rate: $(( (successful_formats * 100) / total_formats ))%" | tee -a "$LOG_FILE"
echo "Complete log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "======================" | tee -a "$LOG_FILE"

# Return overall success/failure
if [ $successful_formats -eq $total_formats ]; then
    echo "All tests passed successfully!"
    exit 0
else
    echo "Some tests failed. Check the log for details."
    exit 1
fi
