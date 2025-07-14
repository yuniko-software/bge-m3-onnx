package com.yunikosoftware.bgem3onnx.performance;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.Map;

/**
 * Container for performance benchmark results
 */
public class PerformanceReport {
    @JsonProperty("test_info")
    private TestInfo testInfo;
    
    @JsonProperty("scenarios")
    private Map<String, BenchmarkResult> scenarios;

    public PerformanceReport() {}

    public PerformanceReport(TestInfo testInfo, Map<String, BenchmarkResult> scenarios) {
        this.testInfo = testInfo;
        this.scenarios = scenarios;
    }

    public TestInfo getTestInfo() { return testInfo; }
    public void setTestInfo(TestInfo testInfo) { this.testInfo = testInfo; }

    public Map<String, BenchmarkResult> getScenarios() { return scenarios; }
    public void setScenarios(Map<String, BenchmarkResult> scenarios) { this.scenarios = scenarios; }

    /**
     * Test information metadata
     */
    public static class TestInfo {
        @JsonProperty("timestamp")
        private String timestamp;
        
        @JsonProperty("test_dataset_size")
        private int testDatasetSize;
        
        @JsonProperty("sample_texts")
        private String[] sampleTexts;

        public TestInfo() {}

        public TestInfo(String timestamp, int testDatasetSize, String[] sampleTexts) {
            this.timestamp = timestamp;
            this.testDatasetSize = testDatasetSize;
            this.sampleTexts = sampleTexts;
        }

        public String getTimestamp() { return timestamp; }
        public void setTimestamp(String timestamp) { this.timestamp = timestamp; }

        public int getTestDatasetSize() { return testDatasetSize; }
        public void setTestDatasetSize(int testDatasetSize) { this.testDatasetSize = testDatasetSize; }

        public String[] getSampleTexts() { return sampleTexts; }
        public void setSampleTexts(String[] sampleTexts) { this.sampleTexts = sampleTexts; }
    }
}