# Booking.com AI Trip Planner - Case Study Implementation

## Overview
This repository contains the implementation and analysis from our case study on **Booking.com's AI-Powered Travel Platform**. This initiative focuses on understanding the AI-driven features that Booking.com built using OpenAI's models, including conversational trip planning, smart filters, property Q&A, and review summarization.

## Resources

### Original Booking.com Resources
- **OpenAI Case Study**: [Booking.com and OpenAI personalize travel at scale](https://openai.com/index/booking-com/)

## Key Concepts from Booking.com's AI Implementation

Booking.com partnered with OpenAI to transform their travel platform with AI-powered features that enhance discovery, personalization, and user experience:

- **AI Trip Planner**: Conversational destination discovery and itinerary building using natural language prompts
- **Smart Filters**: Natural language understanding to map user requests to property filters beyond predefined options
- **Property Q&A**: AI-powered question answering about property details using fine-tuned models on user-generated content
- **AI Review Summaries**: Automated summarization of property reviews into key themes for faster decision-making
- **Help Me Reply**: Automated response generation for guest communications

## Architecture

The implementation includes:
- **Backend**: FastAPI server with semantic search, composite scoring, property Q&A, and itinerary planning
- **Frontend**: React application for the user interface
- **Data Layer**: Qdrant vector database for property embeddings and search
- **AI Integration**: OpenAI models for intent parsing, question answering, and itinerary generation

## Getting Started

### Prerequisites
```bash
# Backend dependencies
pip install -r requirements.txt

# Frontend dependencies
cd frontend
npm install
```

### Get the API Keys

- **OpenAI API Key**: [OpenAI Platform](https://platform.openai.com/api-keys)
- **Qdrant API Key**: [Qdrant Cloud](https://cloud.qdrant.io/)

### Running the Application

```bash
# Start the backend
cd backend
python main.py

# In another terminal, start the frontend
cd frontend
npm run dev
```

## Implementation Notes

This implementation is inspired by Booking.com's approach but adapted to our specific tech stack and use case. The focus is on understanding the core AI integration patterns and conversational travel assistance rather than replicating the exact infrastructure.

## License

This is a case study project for educational purposes only.

## Acknowledgments

- Booking.com and OpenAI for sharing insights on their AI collaboration
- Community contributors to the case study series