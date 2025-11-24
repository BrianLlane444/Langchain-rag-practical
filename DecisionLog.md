## Decision Log (my notes while working)
- Chose Llama 3.2:11b over 70b: Better speed/quality trade-off for M4 Pro
- Selected Redis over PostgreSQL: Better for session/conversation data
- Implemented LangGraph over hardcoded flow: Maintainable MI stages since Keeping MI consistent and not overwhelming the user was a big issue
- Today, langchain depends on langchain-core and often pulls langchain-community. Listing all three isn’t wrong, but it can complicate version resolution.
- Recommendation to pick one approach:

A (simple): keep langchain + langchain-ollama and drop explicit langchain-core / langchain-community, or

B (fine-grained): keep langchain-core + langchain-community + langchain-ollama and drop top-level langchain.


- Downgraded langchain-core because it still works with langchain dependencies while a langgraph package want latest langchain-core. DONE
- may download unstructured
- will keep llama3.1:8b unless good internet for qwen 14b pull
- will upload the other pdfs
- will clean the memory since new embeddings will be created
- ask about knowledge graphs
- ask about change in system prompt template
- ask about chunk details for llama specifically
- ask about redis change
- Take screenshots if somewhat happy with the responses
- Use docker and finish the report

-put docker decision log.md asweell here or in rag_pipeline.