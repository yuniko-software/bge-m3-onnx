#!/bin/bash
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting BGE-M3 ONNX model tests${NC}"

if [ ! -d "onnx" ]; then
    echo -e "${RED}ERROR: onnx directory not found!${NC}"
    echo "Please create an 'onnx' directory in the repository root and add the required ONNX files."
    exit 1
fi

if [ ! -f "onnx/bge_m3_tokenizer.onnx" ]; then
    echo -e "${RED}ERROR: bge_m3_tokenizer.onnx not found!${NC}"
    echo "Please download or generate the BGE-M3 tokenizer ONNX model and place it in the onnx directory."
    exit 1
fi

if [ ! -f "onnx/bge_m3_model.onnx" ]; then
    echo -e "${RED}ERROR: bge_m3_model.onnx not found!${NC}"
    echo "Please download the BGE-M3 model ONNX file and place it in the onnx directory."
    exit 1
fi

# Step 1: Generate reference embeddings using Python
echo -e "${YELLOW}Generating reference embeddings using Python...${NC}"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: python3 command not found!${NC}"
    echo "Please install Python 3 to run this test script."
    exit 1
fi

echo "Found Python: $(python3 --version)"

# Check if required packages are installed
PACKAGES=("onnx" "onnxruntime" "onnxruntime-extensions" "numpy" "pytest" "transformers" "FlagEmbedding", "torch")
MISSING_PACKAGES=()

for pkg in "${PACKAGES[@]}"; do
    # Convert dashes to underscores for import check
    pkg_import=${pkg//-/_}
    if ! python3 -c "import $pkg_import" &> /dev/null; then
        MISSING_PACKAGES+=("$pkg")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo -e "${YELLOW}Installing required Python packages: ${MISSING_PACKAGES[*]}${NC}"
    pip install "${MISSING_PACKAGES[@]}"
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Failed to install required packages${NC}"
        exit 1
    fi
fi

# Run the Python script to generate reference embeddings
pushd samples/python > /dev/null
python3 generate_reference_embeddings.py
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to generate reference embeddings!${NC}"
    popd > /dev/null
    exit 1
fi
popd > /dev/null

echo -e "${GREEN}Reference embeddings generated successfully!${NC}"

# Step 2: Run Python tests
echo -e "${YELLOW}Running Python tests...${NC}"

pushd samples/python > /dev/null
python3 bge_m3_tests.py
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Python tests failed!${NC}"
    popd > /dev/null
    exit 1
fi
popd > /dev/null

echo -e "${GREEN}Python tests passed successfully!${NC}"

# Step 3: Run .NET tests
echo -e "${YELLOW}Running .NET tests...${NC}"

pushd samples/dotnet/BgeM3.Onnx.Tests > /dev/null
dotnet test --verbosity normal

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: .NET tests failed!${NC}"
    popd > /dev/null
    exit 1
fi
popd > /dev/null

echo -e "${GREEN}.NET tests passed successfully!${NC}"

# Step 4: Run Java tests
echo -e "${YELLOW}Running Java tests...${NC}"

# Check if Maven is available
if ! command -v mvn &> /dev/null; then
    echo -e "${RED}ERROR: mvn command not found!${NC}"
    echo "Please install Maven to run Java tests."
    exit 1
fi

pushd samples/java/bge-m3-onnx > /dev/null
mvn test

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Java tests failed!${NC}"
    popd > /dev/null
    exit 1
fi
popd > /dev/null

echo -e "${GREEN}Java tests passed successfully!${NC}"
echo -e "${GREEN}All BGE-M3 tests passed successfully!${NC}"