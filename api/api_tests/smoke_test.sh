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
TEMP_LOG_DIR="$LOG_DIR/temp_logs"
mkdir -p "$TEMP_LOG_DIR"

# Start log file
echo "====== Multimodal Extraction Test - $(date) ======" | tee -a "$LOG_FILE"
echo "Running from base directory: $BASE_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Function to run extraction and log results
run_extraction() {
    local file_ext=$1
    local doc_type=$2
    local test_file="$DATA_DIR/multimodal_test.$file_ext"
    local temp_log="$TEMP_LOG_DIR/${file_ext}_extraction.log"
    local result_file="$TEMP_LOG_DIR/${file_ext}_result"

    # Log start of test
    {
        echo "====== Testing $file_ext extraction ======"
        echo "File: $test_file"
        echo "Document type: $doc_type"
        echo "Started at: $(date)"

        # Check if file exists
        if [ ! -f "$test_file" ]; then
            echo "ERROR: File $test_file not found. Skipping test."
            echo "1" > "$result_file"  # Mark as failed
            echo ""
            return 1
        fi

        # Run extraction
        start_time=$(date +%s)

        nv-ingest-cli --batch_size=$BATCH_SIZE --doc="$test_file" --client_host=$HOST --client_port=$PORT \
          --output_directory="$OUTPUT_DIR" \
          --task="extract:{\"document_type\": \"$doc_type\", $EXTRACT_PARAMS}" --log_level=$LOG_LEVEL \
          2>&1

        exit_code=$?
        end_time=$(date +%s)
        duration=$((end_time - start_time))

        echo "Exit code: $exit_code"
        echo "Duration: $duration seconds"
        echo "Completed at: $(date)"
        echo ""
        echo "--------------------------------------------------"
        echo ""

        # Store exit code in result file
        echo "$exit_code" > "$result_file"
    } > "$temp_log" 2>&1

    # Return exit code for counting
    exit_code=$(cat "$result_file")
    return $exit_code
}

# Counter variables
total_formats=0
successful_formats=0
declare -A test_results=()
declare -A test_durations=()

# Define file formats to test
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

# Launch all extraction tests in parallel
echo "Running extraction tests" | tee -a "$LOG_FILE"
echo "======================" | tee -a "$LOG_FILE"
for ext in "${!formats[@]}"; do
    doc_type="${formats[$ext]}"
    ((total_formats++))

    # Run extraction in background
    run_extraction "$ext" "$doc_type" &
    pid=$!

    # Store PID for tracking
    pids+=($pid)
    format_pids["$pid"]="$ext"
done

# Wait for all processes to complete and collect results
for pid in "${pids[@]}"; do
    wait $pid
    exit_code=$?
    format="${format_pids[$pid]}"

    # Get test duration from log file
    temp_log="$TEMP_LOG_DIR/${format}_extraction.log"
    duration="N/A"
    if [ -f "$temp_log" ]; then
        duration=$(grep "Duration:" "$temp_log" | awk '{print $2}')
        test_durations["$format"]="$duration"
    fi

    # Track results
    if [ $exit_code -eq 0 ]; then
        ((successful_formats++))
        test_results["$format"]="PASS"
    else
        test_results["$format"]="FAIL"

        # For failed tests, add the output to the main log
        echo "" | tee -a "$LOG_FILE"
        echo "====== FAILED: $format extraction ======" | tee -a "$LOG_FILE"
        cat "$temp_log" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
    fi
done

# Print summary
echo "====== SUMMARY ======" | tee -a "$LOG_FILE"
echo "Total formats tested: $total_formats" | tee -a "$LOG_FILE"
echo "Successful extractions: $successful_formats" | tee -a "$LOG_FILE"
echo "Failed extractions: $((total_formats - successful_formats))" | tee -a "$LOG_FILE"
echo "Success rate: $(( (successful_formats * 100) / total_formats ))%" | tee -a "$LOG_FILE"

# Print detailed test results table
echo "" | tee -a "$LOG_FILE"
echo "====== DETAILED RESULTS ======" | tee -a "$LOG_FILE"

# Print table header with proper formatting
printf "%-10s | %-8s | %-15s | %-8s\n" "FORMAT" "RESULT" "DURATION (SEC)" "PID" | tee -a "$LOG_FILE"
printf "%-10s-|-%-8s-|-%-15s-|-%-8s\n" "----------" "--------" "---------------" "--------" | tee -a "$LOG_FILE"

# Print results in tabular format
for pid in "${pids[@]}"; do
    format="${format_pids[$pid]}"
    result="${test_results[$format]:-UNKNOWN}"
    duration="${test_durations[$format]:-N/A}"

    # Color coding for results
    if [ "$result" = "PASS" ]; then
        result_display="\033[0;32mPASS\033[0m"  # Green color
    else
        result_display="\033[0;31mFAIL\033[0m"  # Red color
    fi

    # For console output with colors
    printf "%-10s | %-8s | %-15s | %-8s\n" "$format" "$result" "$duration sec" "$pid" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"

# If there were failures, print failure details
if [ $((total_formats - successful_formats)) -gt 0 ]; then
    echo "====== FAILURE DETAILS ======" | tee -a "$LOG_FILE"
    echo "The following formats failed extraction:" | tee -a "$LOG_FILE"
    for format in "${!test_results[@]}"; do
        if [ "${test_results[$format]}" = "FAIL" ]; then
            echo "- $format" | tee -a "$LOG_FILE"
        fi
    done
    echo "" | tee -a "$LOG_FILE"
    echo "See above for detailed error logs for each failed extraction." | tee -a "$LOG_FILE"
fi

echo "Complete log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "======================" | tee -a "$LOG_FILE"

# Cleanup temp logs
rm -rf "$TEMP_LOG_DIR"

# Return overall success/failure
if [ $successful_formats -eq $total_formats ]; then
    echo "All tests passed successfully!"
    exit 0
else
    echo "Some tests failed. Check the log for details."
    exit 1
fi
