"""
core/test_connections.py
Run this BEFORE starting the system to verify all connections are live.

Usage:
    python core/test_connections.py

Checks:
  1. AgentRouter (Claude Haiku call)
  2. OpenAI Embeddings (text-embedding-3-small)
  3. Tavily Web Search
  4. ChromaDB (local)
  5. WeasyPrint (PDF rendering)
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check(label: str, fn):
    """Run a check and print pass/fail."""
    try:
        result = fn()
        print(f"  ✅  {label}: {result}")
        return True
    except Exception as e:
        print(f"  ❌  {label}: {e}")
        return False


def test_agentrouter():
    from openai import OpenAI
    from core.config import settings, MODELS
    client = OpenAI(
        api_key=settings.agentrouter_api_key,
        base_url=settings.agentrouter_base_url,
    )
    resp = client.chat.completions.create(
        model=MODELS["haiku"],
        messages=[{"role": "user", "content": "Reply with: CONNECTION OK"}],
        max_tokens=20,
        temperature=0,
    )
    text = resp.choices[0].message.content.strip()
    assert "OK" in text.upper() or len(text) > 3
    return f"Model={MODELS['haiku']} | Response='{text[:40]}'"


def test_openai_embeddings():
    from openai import OpenAI
    from core.config import settings
    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input="construction cost estimation test",
    )
    dims = len(resp.data[0].embedding)
    assert dims == 1536
    return f"Embedding dims={dims}"


def test_tavily():
    from tavily import TavilyClient
    from core.config import settings
    client = TavilyClient(api_key=settings.tavily_api_key)
    results = client.search(
        query="construction material prices 2025",
        max_results=1,
    )
    count = len(results.get("results", []))
    assert count >= 1
    return f"Results returned={count}"


def test_chromadb():
    import chromadb
    from core.config import settings
    client = chromadb.PersistentClient(path=settings.chroma_db_path)
    collection = client.get_or_create_collection("connection_test")
    collection.upsert(
        ids=["test_1"],
        documents=["test document for construction estimator"],
        metadatas=[{"type": "test"}],
    )
    result = collection.get(ids=["test_1"])
    assert len(result["ids"]) == 1
    # Cleanup
    client.delete_collection("connection_test")
    return "Local ChromaDB read/write OK"


def test_weasyprint():
    from weasyprint import HTML
    import tempfile, os
    html = "<html><body><h1>WeasyPrint OK</h1></body></html>"
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        tmp_path = f.name
    HTML(string=html).write_pdf(tmp_path)
    size = os.path.getsize(tmp_path)
    os.unlink(tmp_path)
    assert size > 1000
    return f"PDF rendered OK ({size} bytes)"


def main():
    print("\n" + "=" * 60)
    print("  CONSTRUCTION ESTIMATOR — CONNECTION CHECK")
    print("=" * 60)

    results = []
    results.append(check("AgentRouter (Claude Haiku)", test_agentrouter))
    results.append(check("OpenAI Embeddings", test_openai_embeddings))
    results.append(check("Tavily Web Search", test_tavily))
    results.append(check("ChromaDB (local)", test_chromadb))
    results.append(check("WeasyPrint (PDF)", test_weasyprint))

    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"  RESULT: {passed}/{total} checks passed")
    if passed == total:
        print("  🎉  ALL SYSTEMS GO — Ready to build!\n")
        sys.exit(0)
    else:
        print("  ⚠️  Fix the failing checks before proceeding.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
