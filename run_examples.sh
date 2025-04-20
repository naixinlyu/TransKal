#!/bin/bash
# Script to run examples of transformer-based anomaly detection

# Set script to exit on any error
set -e

# Make sure the script is executable
chmod +x examples/simple_example.py
chmod +x examples/custom_example.py
chmod +x examples/tep_example.py
chmod +x run.py

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to run a command and check its result
run_command() {
    echo -e "${BLUE}Running: $1${NC}"
    echo "-------------------------------------"
    if eval $1; then
        echo -e "${GREEN}Command completed successfully!${NC}"
    else
        echo -e "${RED}Command failed with exit code $?${NC}"
        exit 1
    fi
    echo "-------------------------------------"
    echo ""
}

# Show menu
echo -e "${YELLOW}Transformer-Based Anomaly Detection Examples${NC}"
echo "1. Run simple example with synthetic data"
echo "2. Run custom example with your own data"
echo "3. Run TEP dataset example"
echo "4. Help with run.py usage"
echo "0. Exit"
echo ""

# Ask user for choice
read -p "Enter your choice (0-4): " CHOICE

case $CHOICE in
    1)
        echo -e "${YELLOW}Running simple example with synthetic data...${NC}"
        run_command "python examples/simple_example.py"
        ;;
    2)
        echo -e "${YELLOW}Running custom example with your own data...${NC}"
        
        # Ask for data file path
        read -p "Enter path to your data file (CSV): " DATA_FILE
        
        if [ ! -f "$DATA_FILE" ]; then
            echo -e "${RED}Error: File not found: $DATA_FILE${NC}"
            exit 1
        fi
        
        # Ask for basic configuration
        read -p "Enter label column name (default: label, or 'none' if you'll provide fault start): " LABEL_COL
        LABEL_COL=${LABEL_COL:-label}
        
        FAULT_START=""
        if [ "$LABEL_COL" = "none" ]; then
            LABEL_COL=""
            read -p "Enter fault start index: " FAULT_START
            FAULT_START="--fault-start $FAULT_START"
        fi
        
        read -p "Enter time column name (optional): " TIME_COL
        if [ ! -z "$TIME_COL" ]; then
            TIME_COL="--time-col $TIME_COL"
        fi
        
        read -p "Enter feature column names (comma-separated, or leave empty for auto-detection): " FEATURE_COLS
        if [ ! -z "$FEATURE_COLS" ]; then
            FEATURE_COLS="--feature-cols $FEATURE_COLS"
        fi
        
        read -p "Enter model name (default: custom_model): " MODEL_NAME
        MODEL_NAME=${MODEL_NAME:-custom_model}
        
        read -p "Enter number of epochs (default: 30): " EPOCHS
        EPOCHS=${EPOCHS:-30}
        
        read -p "Enter lookback window size (default: 15): " LOOKBACK
        LOOKBACK=${LOOKBACK:-15}
        
        # Build command
        COMMAND="python examples/custom_example.py --data $DATA_FILE"
        
        if [ ! -z "$LABEL_COL" ]; then
            COMMAND="$COMMAND --label-col $LABEL_COL"
        fi
        
        COMMAND="$COMMAND $TIME_COL $FEATURE_COLS $FAULT_START --model-name $MODEL_NAME --epochs $EPOCHS --lookback $LOOKBACK"
        
        # Run the command
        run_command "$COMMAND"
        ;;
    3)
        echo -e "${YELLOW}Running TEP dataset example...${NC}"
        echo "This example requires the TEP dataset files in the current directory:"
        echo "- TEP_FaultFree_Training.RData"
        echo "- TEP_Faulty_Training.RData"
        echo "- TEP_Faulty_Testing.RData"
        
        if [ ! -f "TEP_FaultFree_Training.RData" ] || [ ! -f "TEP_Faulty_Training.RData" ] || [ ! -f "TEP_Faulty_Testing.RData" ]; then
            echo -e "${RED}Error: TEP dataset files not found in the current directory${NC}"
            exit 1
        fi
        
        run_command "python examples/tep_example.py"
        ;;
    4)
        echo -e "${YELLOW}Help with run.py usage:${NC}"
        run_command "python run.py --help"
        ;;
    0)
        echo -e "${YELLOW}Exiting...${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}Done!${NC}"