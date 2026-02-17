#!/bin/bash

# Nextflow Pipeline Runner
# This script helps run the patch_seq_spl Nextflow pipeline

set -e

# Default values
DATASET="sorensen"
PROFILE="standard"
GROUP_BY="three"
RESUME=false
DRY_RUN=false
HELP=false

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run the patch_seq_spl Nextflow pipeline

OPTIONS:
    -d, --dataset DATASET           Dataset name (default: sorensen)
    -p, --profile PROFILE           Execution profile: standard, slurm, niagara, docker, singularity (default: standard)
    -g, --group-by GROUP_BY         Grouping variable: three or five (default: three)
    -r, --resume                    Resume previous run
    -n, --dry-run                   Print command without running
    -h, --help                      Show this help message

EXAMPLES:
    # Run with default settings
    $0

    # Run with SLURM profile
    $0 --profile slurm

    # Run specific dataset with Docker
    $0 --dataset gouwens --profile docker

    # Resume interrupted run
    $0 --resume

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -p|--profile)
            PROFILE="$2"
            shift 2
            ;;
        -g|--group-by)
            GROUP_BY="$2"
            shift 2
            ;;
        -r|--resume)
            RESUME=true
            shift
            ;;
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Check if GENOMIC_DATA_DIR is set
if [[ -z "${GENOMIC_DATA_DIR:-}" ]]; then
    echo -e "${YELLOW}Warning: GENOMIC_DATA_DIR environment variable not set${NC}"
    echo "Please set it before running:"
    echo "  export GENOMIC_DATA_DIR=/path/to/genomic/data"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build command
CMD="nextflow run main.nf"
CMD="$CMD -profile $PROFILE"
CMD="$CMD --dataset $DATASET"
CMD="$CMD --group_by $GROUP_BY"

if [ "$RESUME" = true ]; then
    CMD="$CMD -resume"
fi

# Print command
echo -e "${GREEN}Running:${NC}"
echo "$CMD"
echo ""

# Execute or dry-run
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}[DRY RUN] Command not executed${NC}"
    exit 0
else
    eval "$CMD"
fi
