#!/bin/bash
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting BGE-M3 ONNX Performance Benchmarks${NC}"

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

# Step 1: Generate performance test dataset
echo -e "${YELLOW}Generating performance test dataset...${NC}"

pushd samples/performance_data > /dev/null
python3 generate_dataset.py
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to generate test dataset!${NC}"
    popd > /dev/null
    exit 1
fi
popd > /dev/null

echo -e "${GREEN}Test dataset generated successfully!${NC}"

# Step 2: Run Python performance tests
echo -e "${YELLOW}Running Python performance benchmarks...${NC}"

pushd samples/python > /dev/null
python3 performance_test.py
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Python performance tests failed!${NC}"
    popd > /dev/null
    exit 1
fi
popd > /dev/null

echo -e "${GREEN}Python performance tests completed successfully!${NC}"

# Step 3: Run .NET performance tests
echo -e "${YELLOW}Running .NET performance benchmarks...${NC}"

pushd samples/dotnet/BgeM3.Onnx.Performance > /dev/null
dotnet clean
dotnet run --configuration Release
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: .NET performance tests failed!${NC}"
    popd > /dev/null
    exit 1
fi
popd > /dev/null

echo -e "${GREEN}.NET performance tests completed successfully!${NC}"

# Step 4: Run Java performance tests
echo -e "${YELLOW}Running Java performance benchmarks...${NC}"

# Check if Maven is available
if ! command -v mvn &> /dev/null; then
    echo -e "${RED}ERROR: mvn command not found!${NC}"
    echo "Please install Maven to run Java performance tests."
    exit 1
fi

echo "Found Maven: $(mvn --version | head -n1)"

pushd samples/java/bge-m3-onnx > /dev/null
mvn clean compile exec:java -Pperformance
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Java performance tests failed!${NC}"
    popd > /dev/null
    exit 1
fi
popd > /dev/null

echo -e "${GREEN}Java performance tests completed successfully!${NC}"

# Step 5: Display summary
echo -e "${YELLOW}Performance benchmark results summary:${NC}"

RESULT_FILES=(
    "onnx/performance_python.json"
    "onnx/performance_dotnet.json" 
    "onnx/performance_java.json"
)

for result_file in "${RESULT_FILES[@]}"; do
    if [ -f "$result_file" ]; then
        language=$(basename "$result_file" .json | sed 's/performance_//' | tr '[:lower:]' '[:upper:]')
        echo -e "${CYAN}${language} results: ${result_file}${NC}"
    else
        language=$(basename "$result_file" .json | sed 's/performance_//' | tr '[:lower:]' '[:upper:]')
        echo -e "${RED}WARNING: ${language} results file not found: ${result_file}${NC}"
    fi
done

echo -e "${GREEN}All BGE-M3 performance benchmarks completed successfully!${NC}"
echo "Check the individual JSON files in the onnx directory for detailed results."