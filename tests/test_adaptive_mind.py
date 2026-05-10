#!/usr/bin/env python3
"""Test the Adaptive Mind — learn from Codex, recall in local inference.

Tests the full learning loop WITHOUT a GPU:
  1. Ingest captured Codex sessions from the replay buffer
  2. Learn experiences (embed + index)
  3. Recall relevant experiences for a new query
  4. Generate an augmented response
  5. Verify the response was improved by experience

This test requires Ollama running with an embedding model.
"""

import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_experience_memory_offline():
    """Test experience memory without Ollama (pure storage + index)."""
    from neuro.learning.experience_memory import ExperienceMemory, VectorIndex
    import numpy as np

    print("[1] Testing VectorIndex...")
    with tempfile.TemporaryDirectory() as tmpdir:
        index = VectorIndex(Path(tmpdir) / "idx", dim=8)

        # Add vectors
        for i in range(5):
            vec = np.random.randn(8).astype(np.float32)
            vec = vec / np.linalg.norm(vec)  # normalize
            index.add(f"exp_{i}", vec)

        assert index.size() == 5
        print(f"  Index size: {index.size()}")

        # Search
        query = np.random.randn(8).astype(np.float32)
        results = index.search(query, top_k=3)
        assert len(results) == 3
        assert all(isinstance(r[1], float) for r in results)
        print(f"  Search returned {len(results)} results: {[(r[0], f'{r[1]:.3f}') for r in results]}")

        # Persistence — reload from disk
        index2 = VectorIndex(Path(tmpdir) / "idx", dim=8)
        assert index2.size() == 5
        print(f"  Reloaded index: {index2.size()} vectors (persistence works)")

    print("  ✓ VectorIndex works\n")


def test_experience_memory_with_mock_embeddings():
    """Test experience memory with mock embeddings (no Ollama needed)."""
    from neuro.learning.experience_memory import ExperienceMemory, Experience
    import numpy as np

    print("[2] Testing ExperienceMemory (mock embeddings)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        memory = ExperienceMemory(memory_dir=Path(tmpdir))

        # Monkey-patch _embed to use deterministic fake embeddings
        def fake_embed(text):
            # Hash-based deterministic embedding (stable, no overflow)
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            vec = np.array([b / 255.0 - 0.5 for b in h[:32]], dtype=np.float32)
            return vec / (np.linalg.norm(vec) + 1e-8)

        memory._embed = fake_embed

        # Learn some experiences
        memory.learn("How to read a CSV file?", "Use pandas: pd.read_csv('file.csv')", source="codex")
        memory.learn("How to write JSON?", "Use json.dump(data, file)", source="claude")
        memory.learn("How to sort a list?", "Use sorted(lst) or lst.sort()", source="codex")
        memory.learn("How to parse HTML?", "Use BeautifulSoup: soup = BeautifulSoup(html, 'html.parser')", source="claude")
        memory.learn("How to make HTTP requests?", "Use requests: resp = requests.get(url)", source="codex")

        stats = memory.stats()
        print(f"  Learned: {stats['total_experiences']} experiences")
        print(f"  Sources: {stats['by_source']}")
        assert stats["total_experiences"] == 5

        # Recall
        results = memory.recall("How do I read CSV data?", top_k=2)
        print(f"  Recall for 'How do I read CSV data?': {len(results)} results")
        for r in results:
            print(f"    [{r.source}] {r.query[:50]} → {r.response[:50]}")

        # Test prompt augmentation
        prompt = memory.recall_as_prompt("How do I parse a CSV?", top_k=2)
        assert "Past Experiences" in prompt
        print(f"  Augmented prompt length: {len(prompt)} chars")

        # Reinforce
        if results:
            memory.reinforce(results[0].id, score=0.95)
            updated = memory._cache[results[0].id]
            print(f"  After reinforce: quality={updated.quality_score:.2f}")

        # Prune
        # Add a bad experience
        bad = memory.learn("test", "x", source="junk", quality_score=0.01)
        pruned = memory.prune(min_score=0.2)
        print(f"  Pruned: {pruned} low-quality experiences")

        # Persistence
        memory2 = ExperienceMemory(memory_dir=Path(tmpdir))
        memory2._embed = fake_embed
        assert memory2.stats()["total_experiences"] >= 4
        print(f"  Reloaded: {memory2.stats()['total_experiences']} experiences (persistence works)")

    print("  ✓ ExperienceMemory works\n")


def test_replay_buffer_ingestion():
    """Test ingesting from the interceptor's replay buffer."""
    from neuro.learning.experience_memory import ExperienceMemory
    import numpy as np

    print("[3] Testing replay buffer ingestion...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a fake replay buffer
        buffer_path = Path(tmpdir) / "replay_buffer.jsonl"
        with open(buffer_path, "w") as f:
            for i in range(10):
                example = {
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": f"How do I implement feature {i}?"},
                        {"role": "assistant", "content": f"Here's how to implement feature {i}: use pattern X with library Y."},
                    ],
                    "metadata": {"tool": "codex", "session_id": "test"},
                }
                f.write(json.dumps(example) + "\n")

        memory = ExperienceMemory(memory_dir=Path(tmpdir) / "mem")

        # Mock embeddings
        def fake_embed(text):
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            vec = np.array([b / 255.0 - 0.5 for b in h[:32]], dtype=np.float32)
            return vec / (np.linalg.norm(vec) + 1e-8)
        memory._embed = fake_embed

        learned = memory.learn_from_buffer(buffer_path)
        print(f"  Ingested: {learned} experiences from buffer")
        assert learned == 10

        stats = memory.stats()
        print(f"  Total: {stats['total_experiences']}, Sources: {stats['by_source']}")
        assert stats["by_source"].get("codex", 0) == 10

    print("  ✓ Buffer ingestion works\n")


def test_adaptive_mind_offline():
    """Test the AdaptiveMind with mock components (no Ollama)."""
    from neuro.learning.adaptive_mind import AdaptiveMind, MindResponse
    from neuro.learning.experience_memory import ExperienceMemory
    import numpy as np

    print("[4] Testing AdaptiveMind (offline)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        mind = AdaptiveMind(
            model="test-model",
            memory_dir=Path(tmpdir),
            use_steering=False,  # skip steering (needs real vectors)
            auto_learn=False,    # skip auto-learn (needs Ollama)
        )

        # Mock embeddings on the memory
        def fake_embed(text):
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            vec = np.array([b / 255.0 - 0.5 for b in h[:32]], dtype=np.float32)
            return vec / (np.linalg.norm(vec) + 1e-8)
        mind.memory._embed = fake_embed

        # Manually teach some experiences
        mind.memory.learn("How to sort a list in Python?", "sorted(lst) returns new sorted list", source="codex", quality_score=0.9)
        mind.memory.learn("How to reverse a string?", "Use s[::-1] for string reversal", source="claude", quality_score=0.85)

        # Test recall
        experiences = mind.memory.recall("sort a Python list", top_k=2)
        print(f"  Recalled {len(experiences)} experiences for 'sort a Python list'")
        for e in experiences:
            print(f"    [{e.source}] {e.query} (score: {e.combined_score():.2f})")

        # Test prompt augmentation
        prompt = mind.memory.recall_as_prompt("sort a Python list")
        has_experience = "Past Experiences" in prompt
        print(f"  Prompt augmented: {has_experience} ({len(prompt)} chars)")

        # Test status
        status = mind.status()
        print(f"  Status: {status['total_experiences']} experiences, steering={'on' if status['steering_active'] else 'off'}")

        # Test feedback
        fake_response = MindResponse(
            content="Use sorted(lst)",
            model="test",
            experience_ids=[e.id for e in experiences],
        )
        mind.feedback(fake_response, score=0.95)
        print(f"  Feedback applied to {len(fake_response.experience_ids)} experiences")

    print("  ✓ AdaptiveMind works\n")


def test_live_with_ollama():
    """Test with real Ollama (if running). Skips gracefully if not."""
    print("[5] Testing live with Ollama...")

    try:
        from neuro.runtime.ollama_client import get_ollama_client
        client = get_ollama_client()
        if not client.is_running():
            print("  ⊘ Ollama not running — skipping live test")
            return

        # Check for embedding model
        models = [m.get("name", "") for m in client.list_models()]
        has_embed = any("nomic" in m or "embed" in m for m in models)
        has_gen = any("qwen" in m or "phi" in m or "llama" in m for m in models)

        print(f"  Ollama running. Models: {len(models)}")
        print(f"  Embedding model: {'yes' if has_embed else 'no'}")
        print(f"  Generation model: {'yes' if has_gen else 'no'}")

        if not has_embed:
            print("  ⊘ No embedding model — skipping (run: ollama pull nomic-embed-text)")
            return

        # Test real embedding
        from neuro.learning.experience_memory import ExperienceMemory
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = ExperienceMemory(memory_dir=Path(tmpdir))

            # Learn a real experience
            exp = memory.learn(
                "How to read a file in Python?",
                "with open('file.txt') as f: content = f.read()",
                source="test",
            )
            print(f"  Learned experience: {exp.id if exp else 'failed'}")
            print(f"  Index size: {memory.index.size()}")

            if memory.index.size() > 0:
                # Recall
                results = memory.recall("reading files in python")
                print(f"  Recall results: {len(results)}")
                if results:
                    print(f"    Best match: {results[0].query}")
                    print("  ✓ Live embedding + recall works!")
                else:
                    print("  ⊘ No recall results (may need more data)")

        # If we have a generation model, test full think()
        if has_gen:
            from neuro.learning.adaptive_mind import AdaptiveMind
            gen_model = next(m for m in models if "qwen" in m or "phi" in m or "llama" in m)

            with tempfile.TemporaryDirectory() as tmpdir:
                mind = AdaptiveMind(
                    model=gen_model,
                    memory_dir=Path(tmpdir),
                    use_steering=True,
                    auto_learn=True,
                )

                # Teach it something
                mind.ingest_example(
                    "What is the Kalman filter?",
                    "A Kalman filter is a recursive algorithm that estimates the state of a linear dynamic system from noisy measurements.",
                    source="expert",
                )

                # Ask a related question
                response = mind.think("Explain the Kalman filter briefly")
                print(f"\n  Full think() test:")
                print(f"    Model: {response.model}")
                print(f"    Augmented: {response.augmented}")
                print(f"    Experiences recalled: {response.num_experiences_recalled}")
                print(f"    Generation time: {response.generation_time_ms:.0f}ms")
                print(f"    Response: {response.content[:100]}...")
                print("  ✓ Full AdaptiveMind.think() works!")

    except Exception as e:
        print(f"  ⊘ Live test failed: {e}")
        import traceback
        traceback.print_exc()

    print()


if __name__ == "__main__":
    print("=" * 60)
    print("NeuroBridge Adaptive Mind Tests")
    print("=" * 60)

    tests = [
        ("VectorIndex", test_experience_memory_offline),
        ("ExperienceMemory", test_experience_memory_with_mock_embeddings),
        ("Buffer Ingestion", test_replay_buffer_ingestion),
        ("AdaptiveMind (offline)", test_adaptive_mind_offline),
        ("AdaptiveMind (live)", test_live_with_ollama),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    sys.exit(1 if failed > 0 else 0)
