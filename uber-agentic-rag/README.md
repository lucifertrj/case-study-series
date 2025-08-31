# Uber Genie AI Agent - Case Study Implementation

## Overview
This repository contains the implementation and analysis from our live session on **How Uber Built Their AI Agent (Genie)**. This initiative is focused on understanding the system architecture and approach used by Uber, with our own implementation adapted to our tech stack.

## Resources

### Session Materials

- **YouTube Recording**: [Live Session - How Uber Built Their AI Agent](https://www.youtube.com/watch?v=lROzXRNrXSk)
- **Presentation Slides**: [Google Slides](https://docs.google.com/presentation/d/1sM48hD1S5Pvpd0tjPe6a4oMwa2HfskILTEqvLomuLgM/edit?usp=sharing)

### Original Uber Resources
- **Uber Engineering Blog**: [Enhanced Agentic RAG](https://www.uber.com/en-IN/blog/enhanced-agentic-rag/?uclick_id=9529bd64-1d38-40a6-bc23-88ce151b1384)

## Key Concepts from Uber's Genie

<img width="1536" height="1152" alt="figure-1-17484736625654" src="https://github.com/user-attachments/assets/bcfb6719-030b-46d5-a5be-8e9aae040116" />

Uber's Genie is an enhanced agentic RAG (Retrieval-Augmented Generation) system that helps with intelligent document retrieval and question answering. The system combines:

- **Offline Document Processing**: Includes PDF to Markdown conversation and also the Comparative Study of PDF Parsing experimentation code. 
- **Custom Metadata creation**: For improving the retrieval performance. Also code includes how to use filter condition using Qdrant Vector DB. 
- **RAG orchestration**: Using LangGraph for the RAG orchestration
- **Inference Demo**: Streamlit for the realtime answer generation

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
# Initialize cold start i.e., load the model once so that when to run app, you don't have load model again
python coldstart.py

# Run the main application
python app.py
```

## Implementation Notes

This implementation is inspired by Uber's approach but adapted to our specific needs and tech stack. The focus is on understanding the core concepts and architectural patterns rather than replicating the exact technology choices.

## License

This is a case study project for educational purposes only. 

## Acknowledgments

- Uber Engineering team for sharing their insights on the [Genie AI Agent](https://www.uber.com/en-IN/blog/enhanced-agentic-rag/?uclick_id=9529bd64-1d38-40a6-bc23-88ce151b1384)
- Community contributors to the case study series
