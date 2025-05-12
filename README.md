# Scientific Paper Summarizer
This Python script is designed to load a PDF file of a scientific paper, extract its relevant content, and generate a structured summary using a language model. The script then saves the summary in a Markdown file for easy sharing and review.

## Features
- **PDF Extraction:** The script extracts content from PDF files using the PyPDFLoader from the langchain library.
- **Reference Removal:** It filters out references, acknowledgments, and other irrelevant sections to focus on the core scientific content.
- **AI-Generated Summary:** The script uses the OllamaLLM model to generate a concise, clear summary of the scientific paper.
- **Metadata Handling:** Metadata such as title, author, year of publication, and keywords are extracted from the PDF and included in the final Markdown file.
- **Markdown Output:** The generated summary is saved as a .md file, which includes metadata and the full summary.

## Required Python libraries:
- re
- datetime
- langchain_core
- langchain_ollama
- langchain_community

## The Output Should Look Something Like This:
![file_order](https://github.com/Fran-bot0/Scientific-Article-Summary-Maker/blob/main/output/output.png)
