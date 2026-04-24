import cognee


class CogneeService:
    """Service for interacting with Cognee AI memory."""

    @staticmethod
    async def add_and_cognify(text: str) -> None:
        """Add text to Cognee memory and process it."""
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        await cognee.add(text)
        await cognee.cognify()

    @staticmethod
    async def search(query: str) -> str:
        """Search Cognee memory and return formatted context."""
        results = await cognee.search(query_text=query)

        memory_context = ""
        if results:
            for result in results:
                if isinstance(result, dict) and "search_result" in result:
                    memory_context += "\n".join(result["search_result"]) + "\n"
                elif hasattr(result, "search_result"):
                    memory_context += "\n".join(result.search_result) + "\n"

        if not memory_context.strip():
            memory_context = "No specific information found in memory for this query."

        return memory_context

    @staticmethod
    async def reset_memory() -> None:
        """Clear all Cognee memory."""
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
