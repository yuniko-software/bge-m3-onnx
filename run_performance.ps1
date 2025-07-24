function Write-Green {
    param([string]$Text)
    Write-Host $Text -ForegroundColor Green
}

function Write-Red {
    param([string]$Text)
    Write-Host $Text -ForegroundColor Red
}

function Write-Yellow {
    param([string]$Text)
    Write-Host $Text -ForegroundColor Yellow
}

Write-Yellow "Starting BGE-M3 ONNX Performance Benchmarks"

if (-not (Test-Path "onnx")) {
    Write-Red "ERROR: onnx directory not found!"
    Write-Host "Please create an 'onnx' directory in the repository root and add the required ONNX files."
    exit 1
}

if (-not (Test-Path "onnx/bge_m3_tokenizer.onnx")) {
    Write-Red "ERROR: bge_m3_tokenizer.onnx not found!"
    Write-Host "Please download or generate the BGE-M3 tokenizer ONNX model and place it in the onnx directory."
    exit 1
}

if (-not (Test-Path "onnx/bge_m3_model.onnx")) {
    Write-Red "ERROR: bge_m3_model.onnx not found!"
    Write-Host "Please download the BGE-M3 model ONNX file and place it in the onnx directory."
    exit 1
}

# Step 1: Generate performance test dataset
Write-Yellow "Generating performance test dataset..."

Push-Location "samples\performance_data"
try {
    python generate_dataset.py
    if ($LASTEXITCODE -ne 0) {
        Write-Red "ERROR: Failed to generate test dataset!"
        exit 1
    }
} catch {
    Write-Red "ERROR: Failed to generate test dataset!"
    Write-Host $_.Exception.Message
    exit 1
} finally {
    Pop-Location
}

Write-Green "Test dataset generated successfully!"

# Step 2: Run Python performance tests
Write-Yellow "Running Python performance benchmarks..."

Push-Location "samples\python"
try {
    python performance_test.py
    if ($LASTEXITCODE -ne 0) {
        Write-Red "ERROR: Python performance tests failed!"
        exit 1
    }
} catch {
    Write-Red "ERROR: Python performance tests failed!"
    Write-Host $_.Exception.Message
    exit 1
} finally {
    Pop-Location
}

Write-Green "Python performance tests completed successfully!"

# Step 3: Run .NET performance tests
Write-Yellow "Running .NET performance benchmarks..."

Push-Location "samples\dotnet\BgeM3.Onnx.Performance"
try {
    dotnet run --configuration Release
    if ($LASTEXITCODE -ne 0) {
        Write-Red "ERROR: .NET performance tests failed!"
        exit 1
    }
} catch {
    Write-Red "ERROR: .NET performance tests failed!"
    Write-Host $_.Exception.Message
    exit 1
} finally {
    Pop-Location
}

Write-Green ".NET performance tests completed successfully!"

# Step 4: Run Java performance tests
Write-Yellow "Running Java performance benchmarks..."

# Check if Maven is available
try {
    $mavenVersion = mvn --version
    Write-Host "Found Maven: $($mavenVersion -split "`n" | Select-Object -First 1)"
} catch {
    Write-Red "ERROR: Maven command not found!"
    Write-Host "Please install Maven to run Java performance tests."
    exit 1
}

Push-Location "samples\java\bge-m3-onnx"
try {
    mvn clean compile exec:java -Pperformance
    if ($LASTEXITCODE -ne 0) {
        Write-Red "ERROR: Java performance tests failed!"
        exit 1
    }
} catch {
    Write-Red "ERROR: Java performance tests failed!"
    Write-Host $_.Exception.Message
    exit 1
} finally {
    Pop-Location
}

Write-Green "Java performance tests completed successfully!"

# Step 5: Display summary
Write-Yellow "Performance benchmark results summary:"

$resultFiles = @(
    "onnx\performance_python.json",
    "onnx\performance_dotnet.json", 
    "onnx\performance_java.json"
)

foreach ($resultFile in $resultFiles) {
    if (Test-Path $resultFile) {
        $language = ($resultFile -split "_")[-1] -replace "\.json", ""
        Write-Host "$($language.ToUpper()) results: $resultFile" -ForegroundColor Cyan
    } else {
        $language = ($resultFile -split "_")[-1] -replace "\.json", ""
        Write-Red "WARNING: $($language.ToUpper()) results file not found: $resultFile"
    }
}

Write-Green "All BGE-M3 performance benchmarks completed successfully!"
Write-Host "Check the individual JSON files in the onnx directory for detailed results."