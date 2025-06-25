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

Write-Yellow "Starting BGE-M3 ONNX model tests"

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

# Step 1: Generate reference embeddings using Python
Write-Yellow "Generating reference embeddings using Python..."

# Check if Python is available
try {
    $pythonVersion = python --version
    Write-Host "Found Python: $pythonVersion"
} catch {
    Write-Red "ERROR: Python command not found!"
    Write-Host "Please install Python to run this test script."
    exit 1
}

# Check if required packages are installed
$packages = @("onnx" "onnxruntime", "onnxruntime_extensions", "numpy")
$missingPackages = @()

foreach ($pkg in $packages) {
    $importCheck = python -c "try:
    import $($pkg.Replace('-', '_'))
    print('OK')
except ImportError:
    print('Missing')"
    
    if ($importCheck -match "Missing") {
        $pipPkg = $pkg -replace "_", "-"
        $missingPackages += $pipPkg
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Yellow "Installing required Python packages: $($missingPackages -join ', ')"
    foreach ($pkg in $missingPackages) {
        pip install $pkg
        if ($LASTEXITCODE -ne 0) {
            Write-Red "ERROR: Failed to install package $pkg"
            exit 1
        }
    }
}

# Run the Python script to generate reference embeddings
try {
    python generate_reference_embeddings.py
    if ($LASTEXITCODE -ne 0) {
        Write-Red "ERROR: Failed to generate reference embeddings!"
        exit 1
    }
} catch {
    Write-Red "ERROR: Failed to generate reference embeddings!"
    Write-Host $_.Exception.Message
    exit 1
}

Write-Green "Reference embeddings generated successfully!"

# Step 2: Run .NET tests
Write-Yellow "Running .NET tests..."

Push-Location "samples\dotnet\BgeM3.Onnx.Tests"
try {
    dotnet test --verbosity normal
    if ($LASTEXITCODE -ne 0) {
        Write-Red "ERROR: .NET tests failed!"
        exit 1
    }
} catch {
    Write-Red "ERROR: .NET tests failed!"
    Write-Host $_.Exception.Message
    exit 1
} finally {
    Pop-Location
}

Write-Green ".NET tests passed successfully!"

# Step 3: Run Java tests
Write-Yellow "Running Java tests..."

# Check if Maven is available
try {
    $mavenVersion = mvn --version
    Write-Host "Found Maven: $($mavenVersion -split "`n" | Select-Object -First 1)"
} catch {
    Write-Red "ERROR: Maven command not found!"
    Write-Host "Please install Maven to run Java tests."
    exit 1
}

Push-Location "samples\java\bge-m3-onnx"
try {
    mvn test
    if ($LASTEXITCODE -ne 0) {
        Write-Red "ERROR: Java tests failed!"
        exit 1
    }
} catch {
    Write-Red "ERROR: Java tests failed!"
    Write-Host $_.Exception.Message
    exit 1
} finally {
    Pop-Location
}

Write-Green "Java tests passed successfully!"
Write-Green "All BGE-M3 tests passed successfully!"