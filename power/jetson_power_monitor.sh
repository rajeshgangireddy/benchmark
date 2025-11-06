#!/bin/bash

# ============================================================================
# Jetson Orin Power Monitoring Script
# ============================================================================
# Monitors VDD_GPU_SOC, VDD_CPU_CV, and VIN_SYS_5V0 power rails using tegrastats
# 
# Usage: ./jetson_power_monitor.sh [output_file] [interval_ms]
#
# Arguments:
#   output_file  - CSV file to save power measurements (default: power_log.csv)
#   interval_ms  - Sampling interval in milliseconds (default: 1000)
#
# Examples:
#   ./jetson_power_monitor.sh                      # Use defaults
#   ./jetson_power_monitor.sh my_power.csv         # Custom output file
#   ./jetson_power_monitor.sh my_power.csv 500     # Sample every 500ms
#
# Output: CSV file with timestamp, total power, and per-rail breakdown
# On Exit (Ctrl+C): Displays summary statistics
# ============================================================================

# Configuration
OUTPUT_FILE="${1:-power_log.csv}"
INTERVAL_MS="${2:-1000}"
TEMP_DATA="/tmp/power_measurements_$$.txt"

# Handle Ctrl+C (SIGINT) to print summary and exit cleanly
trap cleanup INT TERM

cleanup() {
    echo -e '\n\n=========================================='
    echo 'Power Measurement Summary'
    echo '=========================================='
    
    if [[ -f "$TEMP_DATA" && -s "$TEMP_DATA" ]]; then
        awk -v interval="$INTERVAL_MS" '
        BEGIN {
            min_current = 999999; max_current = 0; sum_current = 0;
            min_avg = 999999; max_avg = 0; sum_avg = 0;
            count = 0;
        }
        {
            current = $1;
            avg = $2;
            
            if (current < min_current) min_current = current;
            if (current > max_current) max_current = current;
            sum_current += current;
            
            if (avg < min_avg) min_avg = avg;
            if (avg > max_avg) max_avg = avg;
            sum_avg += avg;
            
            count++;
        }
        END {
            if (count > 0) {
                printf "Duration: %.2f seconds (%.2f minutes)\n", count * interval / 1000, count * interval / 60000;
                printf "Samples:  %d\n", count;
                printf "\nTotal Current Power:\n";
                printf "  Min:     %d mW (%.3f W)\n", min_current, min_current/1000;
                printf "  Max:     %d mW (%.3f W)\n", max_current, max_current/1000;
                printf "  Average: %d mW (%.3f W)\n", sum_current/count, sum_current/count/1000;
                printf "\nTotal Average Power:\n";
                printf "  Min:     %d mW (%.3f W)\n", min_avg, min_avg/1000;
                printf "  Max:     %d mW (%.3f W)\n", max_avg, max_avg/1000;
                printf "  Average: %d mW (%.3f W)\n", sum_avg/count, sum_avg/count/1000;
                
                total_energy_j = (sum_current / count) * (count * interval / 1000) / 1000;
                printf "\nTotal Energy Consumed:\n";
                printf "  %.2f J (%.6f Wh)\n", total_energy_j, total_energy_j/3600;
            }
        }
        ' "$TEMP_DATA"
    else
        echo "No data collected."
    fi
    
    echo -e "\nData saved to: $OUTPUT_FILE"
    echo "=========================================="
    rm -f "$TEMP_DATA"
    exit 0
}

# Check if tegrastats is available
if ! command -v tegrastats &> /dev/null; then
    echo "ERROR: tegrastats not found. This script requires a Jetson device."
    exit 1
fi

# Initialize CSV file with header
echo "timestamp,total_current_mW,total_avg_mW,gpu_soc_current_mW,gpu_soc_avg_mW,cpu_cv_current_mW,cpu_cv_avg_mW,sys_5v_current_mW,sys_5v_avg_mW" > "$OUTPUT_FILE"

echo "=========================================="
echo "Jetson Power Monitor"
echo "=========================================="
echo "Output:   $OUTPUT_FILE"
echo "Interval: ${INTERVAL_MS} ms"
echo "Rails:    VDD_GPU_SOC, VDD_CPU_CV, VIN_SYS_5V0"
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

# Start tegrastats and process output
tegrastats --interval "$INTERVAL_MS" | while read -r line; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S.%3N')
    
    # Initialize power sums
    current_power_sum_mW=0
    average_power_sum_mW=0
    
    # Individual rail measurements
    declare -A current_vals
    declare -A average_vals
    
    # Parse the line
    IFS=' ' read -ra fields <<< "$line"
    
    for ((i=0; i<${#fields[@]}; i++)); do
        if [[ "${fields[i]}" == "VDD_GPU_SOC" || "${fields[i]}" == "VDD_CPU_CV" || "${fields[i]}" == "VIN_SYS_5V0" ]]; then
            label="${fields[i]}"
            power_string="${fields[i+1]}"
            
            # Parse current/average (e.g., "5582mW/5583mW")
            IFS='/' read -ra values <<< "$power_string"
            current_mW="${values[0]%mW}"
            avg_mW="${values[1]%mW}"
            
            # Store values
            current_vals["$label"]="$current_mW"
            average_vals["$label"]="$avg_mW"
            
            # Sum totals
            current_power_sum_mW=$((current_power_sum_mW + current_mW))
            average_power_sum_mW=$((average_power_sum_mW + avg_mW))
        fi
    done
    
    # Print to console
    printf "[%s] Total: %5d mW (%.3f W) | Avg: %5d mW (%.3f W)\n" \
        "$timestamp" \
        "$current_power_sum_mW" \
        "$(awk "BEGIN {printf \"%.3f\", $current_power_sum_mW/1000}")" \
        "$average_power_sum_mW" \
        "$(awk "BEGIN {printf \"%.3f\", $average_power_sum_mW/1000}")"
    
    # Save to CSV
    echo "$timestamp,$current_power_sum_mW,$average_power_sum_mW,${current_vals[VDD_GPU_SOC]:-0},${average_vals[VDD_GPU_SOC]:-0},${current_vals[VDD_CPU_CV]:-0},${average_vals[VDD_CPU_CV]:-0},${current_vals[VIN_SYS_5V0]:-0},${average_vals[VIN_SYS_5V0]:-0}" >> "$OUTPUT_FILE"
    
    # Save for statistics
    echo "$current_power_sum_mW $average_power_sum_mW" >> "$TEMP_DATA"
done
