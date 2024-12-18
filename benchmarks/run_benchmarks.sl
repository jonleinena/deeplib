#!/bin/bash
#SBATCH --job-name=IKUN_test_paligemma
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jleinena@vicomtech.org     # Where to send mail
#SBATCH --ntasks=1        # Run on a single CPU
#SBATCH --time=72:00:00    # Time limit hrs:min:sec
#SBATCH --partition=gpu    # Mandatory for GPU works
#SBATCH --mem=64GB        # Job memory request
#SBATCH --gres=gpu:1      # GPU number
#SBATCH --output=ikun_vit_%j.log   # Standard output and error log
#SBATCH --nodes=1    # Number of nodes

# Stop on any error
set -e

# Get the directory where the script is located
PROJECT_ROOT="/gpfs/VICOMTECH/proiektuak/DI11/IKUN/deeplib"

# Create necessary directories
mkdir -p "${PROJECT_ROOT}/data/pascal_voc"
mkdir -p "${PROJECT_ROOT}/data/cityscapes"
mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/benchmark_results"

# Function to check if CUDA is available
check_cuda() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "WARNING: NVIDIA driver not found. Running on CPU."
        DEVICE="--device cpu"
    else
        echo "CUDA available. Running on GPU."
        DEVICE=""
    fi
}

# Function to check dataset existence
check_dataset() {
    local dataset=$1
    local data_dir=$2
    
    case $dataset in
        "pascal_voc")
            if [ ! -d "${data_dir}/VOCdevkit" ]; then
                echo "Pascal VOC dataset not found. Will be downloaded automatically during benchmark."
            else
                echo "Pascal VOC dataset found."
            fi
            ;;
        "cityscapes")
            if [ ! -d "${data_dir}/leftImg8bit" ] || [ ! -d "${data_dir}/gtFine" ]; then
                echo "ERROR: Cityscapes dataset not found in ${data_dir}"
                echo "Please download from https://www.cityscapes-dataset.com/ and place:"
                echo "- leftImg8bit/ directory in ${data_dir}/leftImg8bit"
                echo "- gtFine/ directory in ${data_dir}/gtFine"
                exit 1
            else
                echo "Cityscapes dataset found."
            fi
            ;;
    esac
}

# Function to cleanup datasets
cleanup_datasets() {
    local keep_cityscapes=$1  # Pass 1 to keep Cityscapes, 0 to remove it
    
    echo "Cleaning up datasets..."
    
    # Always remove Pascal VOC as it can be downloaded again
    if [ -d "${PROJECT_ROOT}/data/pascal_voc" ]; then
        echo "Removing Pascal VOC dataset..."
        rm -rf "${PROJECT_ROOT}/data/pascal_voc"
    fi
    
    # Only remove Cityscapes if specified (since it requires manual download)
    if [ "$keep_cityscapes" -eq 0 ] && [ -d "${PROJECT_ROOT}/data/cityscapes" ]; then
        echo "Removing Cityscapes dataset..."
        rm -rf "${PROJECT_ROOT}/data/cityscapes"
    fi
    
    echo "Cleanup completed."
}

# Function to handle cleanup on script exit
cleanup_on_exit() {
    local exit_code=$?
    
    # Create the summary before cleanup
    echo "Creating benchmark summary..."
    python - << EOF
import yaml
from pathlib import Path
import glob
import datetime

def create_summary():
    results_dir = Path("${PROJECT_ROOT}/benchmark_results")
    summary = {
        'timestamp': datetime.datetime.now().isoformat(),
        'system_info': {
            'python_version': '$(python --version 2>&1)',
            'cuda_available': '$(nvidia-smi -L 2>/dev/null || echo "No")',
        },
        'benchmarks': {}
    }
    
    # Collect all benchmark_summary.yaml files
    for summary_file in results_dir.glob('**/benchmark_summary.yaml'):
        with open(summary_file, 'r') as f:
            data = yaml.safe_load(f)
            dataset = summary_file.parent.parent.name
            if dataset not in summary['benchmarks']:
                summary['benchmarks'][dataset] = {}
            summary['benchmarks'][dataset].update(data)
    
    # Save complete summary
    with open(results_dir / 'complete_benchmark_summary.yaml', 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)

create_summary()
EOF
    
    echo "Complete benchmark summary saved to ${PROJECT_ROOT}/benchmark_results/complete_benchmark_summary.yaml"
    
    # Cleanup datasets
    cleanup_datasets 0  # 0 means remove all datasets including Cityscapes
    
    exit $exit_code
}

# Register cleanup function to run on script exit
trap cleanup_on_exit EXIT

# Function to run benchmark
run_benchmark() {
    local dataset=$1
    local model=$2
    local data_dir="${PROJECT_ROOT}/data/${dataset}"
    
    echo "Running benchmark for ${dataset} ${model:+with model $model}..."
    
    # Check dataset
    check_dataset "$dataset" "$data_dir"
    
    # Construct command
    cmd="python3 ${PROJECT_ROOT}/benchmarks/segmentation_benchmarks.py \
        --dataset ${dataset} \
        --data_dir ${data_dir} \
        --output_dir ${PROJECT_ROOT}/benchmark_results \
        --batch_size 16 \
        --epochs 50 \
        ${DEVICE}"
    
    # Add model if specified
    if [ ! -z "$model" ]; then
        cmd="${cmd} --model ${model}"
    fi
    
    # Run benchmark
    echo "Executing: $cmd"
    eval $cmd
}

# Main execution

# Check CUDA availability
check_cuda

# Print system info
echo "=== System Information ==="
echo "Python version: $(python --version 2>&1)"
echo "CUDA devices: $(nvidia-smi -L 2>/dev/null || echo 'No CUDA devices found')"
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | awk '/^Mem:/{print $2}')"
echo "========================="

# Ensure we're in the project root
cd "${PROJECT_ROOT}"

source activate deeplib
# Install/update dependencies
echo "Installing/updating dependencies..."
pip install -e .[all]

# Run benchmarks
echo "Starting benchmarks..."

# Pascal VOC benchmarks
run_benchmark "pascal_voc"

# Cityscapes benchmarks (if available)
if [ -d "${PROJECT_ROOT}/data/cityscapes/leftImg8bit" ]; then
    run_benchmark "cityscapes"
else
    echo "Skipping Cityscapes benchmarks (dataset not found)"
fi

# Optional: Run specific model benchmarks
# Uncomment and modify as needed:
# run_benchmark "pascal_voc" "DeepLabV3Plus"
# run_benchmark "cityscapes" "UNet"

echo "Benchmarking completed!"
echo "Results are available in ${PROJECT_ROOT}/benchmark_results/"
echo "To view TensorBoard logs, run:"
echo "tensorboard --logdir ${PROJECT_ROOT}/benchmark_results"

# Note: Cleanup will be handled by the cleanup_on_exit function registered with trap 