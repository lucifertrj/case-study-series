# Case-study-series

> Engineering Blog article prototypes

## Use Cases

| Use Case | Description | Folder | Article URL |
|----------|-------------|--------|-------------|
| Uber Genie AI Agent | Enhanced Agentic RAG system implementation inspired by Uber's approach. Includes offline document processing, custom metadata creation, RAG orchestration using LangGraph, and real-time inference demo with Streamlit. | [uber-agentic-rag](./uber-agentic-rag) | [Enhanced Agentic RAG](https://www.uber.com/en-IN/blog/enhanced-agentic-rag/?uclick_id=9529bd64-1d38-40a6-bc23-88ce151b1384) |
| Booking.com AI Trip Planner | AI-powered travel platform with conversational trip planning, smart filters, property Q&A, and review summarization inspired by Booking.com's OpenAI integration. | [bookingcom-trip-planner](./bookingcom-trip-planner) | [Booking.com AI Case Study](https://openai.com/index/booking-com/) |
| RFP Offer Analyst | PoC app for evaluating supplier offers against an RFP: upload the RFP and offers, review or add grading criteria, run LLM-backed grading across every offer/criterion cell, rank suppliers with evidence, and chat with an individual offer via payload-filtered vector retrieval, inspired by BMW Group's "Offer Analyst" procurement solution. | [rfp-offer-analyst](./rfp-offer-analyst) | [Revamping procurement operations with generative AI](https://aws.amazon.com/blogs/industries/revamping-procurement-operations-with-generative-ai/) |
| Airbnb Eval-Driven Development | Support-policy RAG assistant with a three-layer evaluation stack (programmatic checks, TruLens LLM-as-judge evaluators, and human calibration with Cohen's kappa), Git-enforced experiment gates, and production traffic monitoring, inspired by Airbnb's eval-driven development approach. | [airbnb-eval-driven-development](./airbnb-eval-driven-development) | [Eval-driven development: Lessons from evaluating GenAI at scale](https://medium.com/airbnb-engineering/eval-driven-development-lessons-from-evaluating-genai-at-scale-e817e5ae5788) |

👉 Disclaimer:

This initiative is purely to understand how the system and approach work. The tech stack we use doesn’t have to be the same as what Uber used. The goal is just to understand the flow and try to build something around it. 