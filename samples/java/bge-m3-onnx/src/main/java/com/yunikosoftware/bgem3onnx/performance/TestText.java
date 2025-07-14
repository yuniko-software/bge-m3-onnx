package com.yunikosoftware.bgem3onnx.performance;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Test text data structure from the performance dataset
 */
public class TestText {
    @JsonProperty("id")
    private int id;
    
    @JsonProperty("text")
    private String text;
    
    @JsonProperty("length_category")
    private String lengthCategory;
    
    @JsonProperty("language")
    private String language;
    
    @JsonProperty("domain")
    private String domain;
    
    @JsonProperty("word_count")
    private int wordCount;
    
    @JsonProperty("char_count")
    private int charCount;
    
    @JsonProperty("source")
    private String source;

    // Default constructor for Jackson
    public TestText() {}

    public TestText(int id, String text, String lengthCategory, String language, 
                   String domain, int wordCount, int charCount, String source) {
        this.id = id;
        this.text = text;
        this.lengthCategory = lengthCategory;
        this.language = language;
        this.domain = domain;
        this.wordCount = wordCount;
        this.charCount = charCount;
        this.source = source;
    }

    // Getters and setters
    public int getId() { return id; }
    public void setId(int id) { this.id = id; }

    public String getText() { return text; }
    public void setText(String text) { this.text = text; }

    public String getLengthCategory() { return lengthCategory; }
    public void setLengthCategory(String lengthCategory) { this.lengthCategory = lengthCategory; }

    public String getLanguage() { return language; }
    public void setLanguage(String language) { this.language = language; }

    public String getDomain() { return domain; }
    public void setDomain(String domain) { this.domain = domain; }

    public int getWordCount() { return wordCount; }
    public void setWordCount(int wordCount) { this.wordCount = wordCount; }

    public int getCharCount() { return charCount; }
    public void setCharCount(int charCount) { this.charCount = charCount; }

    public String getSource() { return source; }
    public void setSource(String source) { this.source = source; }
}