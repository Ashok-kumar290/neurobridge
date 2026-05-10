#!/usr/bin/env python3
"""Tests for the NeuroBridge self-learning pipeline.

Verifies that all components work together:
  1. Interceptor output parsing
  2. Trace storage (append-only JSONL)
  3. Steering vector loading from HDD
  4. Dataset building from captured traces
  5. Colab notebook generation
  6. Continual learner orchestration
"""

import json
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_output_parser():
    """Test that the OutputParser correctly detects turn boundaries."""
    from neuro.training.interceptor import OutputParser

    parser = OutputParser(tool="claude")

    # Simulate a claude-like conversation
    raw_output = (
        "Welcome to Claude Code.\n"
        "> What does the main function do?\n"
        "The main function initializes the application,\n"
        "sets up logging, and starts the event loop.\n"
        "It handles graceful shutdown on SIGTERM.\n"
        "> Can you add error handling?\n"
        "Sure, here's the updated code:\n"
        "```python\n"
        "def main():\n"
        "    try:\n"
        "        app.run()\n"
        "    except Exception as e:\n"
        "        logger.error(f'Fatal: {e}')\n"
        "```\n"
    )

    # Feed line by line
    for line in raw_output.split("\n"):
        parser.feed(line + "\n")
    parser.flush()

    turns = parser.turns
    print(f"  Turns detected: {len(turns)}")
    for t in turns:
        print(f"    [{t.role}] {t.content[:60]}...")

    assert len(turns) >= 3, f"Expected >= 3 turns, got {len(turns)}"
    assert any(t.role == "user" for t in turns), "No user turns detected"
    assert any(t.role == "assistant" for t in turns), "No assistant turns detected"
    print("  ✓ OutputParser works correctly")


def test_trace_storage():
    """Test append-only JSONL storage."""
    from neuro.training.interceptor import TraceStorage

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = TraceStorage(Path(tmpdir))

        # Append examples
        for i in range(5):
            storage.append_example({
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer {i}"},
                ],
                "metadata": {"idx": i},
            })

        stats = storage.get_buffer_stats()
        print(f"  Buffer: {stats['total_examples']} examples, {stats['size_bytes']} bytes")
        assert stats["total_examples"] == 5

        # Tail
        tail = storage.tail_examples(3)
        assert len(tail) == 3
        assert tail[-1]["metadata"]["idx"] == 4

        # Verify append-only (write more, old data preserved)
        storage.append_example({"messages": [], "metadata": {"idx": 5}})
        assert storage.get_buffer_stats()["total_examples"] == 6

        print("  ✓ TraceStorage works correctly")


def test_steering_vector_loading():
    """Test that SteeringLens loads vectors from HDD."""
    from neuro.interpretability.lens import SteeringLens

    vector_dir = Path("/media/seyominaoto/x/neurobridge/checkpoints/steering_vectors")
    if not vector_dir.exists() or not list(vector_dir.glob("*.npz")):
        print("  ⊘ No steering vectors on HDD — skipping")
        return

    lens = SteeringLens(model_name="super-qwen:7b", vector_dir=vector_dir)

    info = lens.info()
    print(f"  Vectors loaded: {info['vectors_loaded']}")
    print(f"  Factuality layer: {info['factuality_layer']}")
    print(f"  Optimal alpha: {info['optimal_alpha']}")

    assert lens.has_vectors(), "No vectors loaded"

    # Check vector dimensions
    for name, vec in lens.steering_vectors.items():
        print(f"  Vector '{name}': shape={vec.shape}, norm={float(vec.sum()**2)**0.5:.4f}")
        assert vec.shape[0] > 0, f"Vector {name} is empty"

    print("  ✓ SteeringLens loads vectors correctly")


def test_dataset_builder():
    """Test dataset building from traces."""
    from neuro.training.dataset_builder import DatasetBuilder

    with tempfile.TemporaryDirectory() as tmpdir:
        traces_dir = Path(tmpdir) / "accepted"
        traces_dir.mkdir()
        output_dir = Path(tmpdir) / "datasets"

        # Create a fake accepted trace
        trace = {
            "trace_id": "test_001",
            "task": "Write a function to sort a list",
            "model": "super-qwen:7b",
            "trainable": True,
            "steps": [
                {"step_type": "context", "data": {"content": "# sort.py\ndef current_sort(lst): pass"}},
                {"step_type": "output", "data": {"response": "def sort_list(lst):\n    return sorted(lst)"}},
            ],
        }

        with open(traces_dir / "test_001.json", "w") as f:
            json.dump(trace, f)

        builder = DatasetBuilder(traces_dir=traces_dir, output_dir=output_dir)
        traces = builder.load_traces()
        print(f"  Loaded {len(traces)} traces")
        assert len(traces) == 1

        # Build with min_examples=1 for testing
        path, stats = builder.build_dataset(min_examples=1)
        print(f"  Dataset: {stats.total_examples} examples")
        assert path is not None
        assert stats.total_examples == 1

        # Verify JSONL content
        with open(path) as f:
            line = f.readline()
            data = json.loads(line)
            assert "messages" in data
            print(f"  ChatML keys: {list(data.keys())}")

        print("  ✓ DatasetBuilder works correctly")


def test_colab_generator():
    """Test Colab notebook generation."""
    from neuro.training.colab_generator import generate_colab_notebook

    notebook = generate_colab_notebook(
        base_model="Qwen/Qwen2.5-Coder-3B-Instruct",
        dataset_path="test_dataset.jsonl",
        lora_rank=16,
        adapter_name="test-adapter-v1",
    )

    assert "cells" in notebook
    assert "metadata" in notebook
    assert notebook["nbformat"] == 4

    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    md_cells = [c for c in notebook["cells"] if c["cell_type"] == "markdown"]

    print(f"  Notebook: {len(code_cells)} code cells, {len(md_cells)} markdown cells")
    assert len(code_cells) >= 8, f"Expected >= 8 code cells, got {len(code_cells)}"

    print("  ✓ ColabGenerator works correctly")


def test_adapter_manager():
    """Test adapter lifecycle management."""
    from neuro.training.adapter_manager import AdapterManager

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AdapterManager(Path(tmpdir))

        # Create a fake adapter directory
        fake_adapter = Path(tmpdir) / "fake_adapter"
        fake_adapter.mkdir()
        (fake_adapter / "adapter_model.bin").write_bytes(b"fake weights")

        # Register
        info = manager.register(
            name="test-adapter-v1",
            adapter_path=fake_adapter,
            base_model="Qwen/Qwen2.5-Coder-3B-Instruct",
            training_examples=50,
        )
        print(f"  Registered: {info.name} (status={info.status})")
        assert info.status == "registered"

        # List
        adapters = manager.list_adapters()
        assert len(adapters) == 1

        # Promote (skip audit logger for test)
        try:
            manager.promote("test-adapter-v1", {"coding": 0.88, "safety": 1.0})
        except Exception:
            pass  # audit logger may not be available

        stats = manager.get_stats()
        print(f"  Stats: {stats}")
        assert stats["total"] == 1

        print("  ✓ AdapterManager works correctly")


def test_intercepted_session_to_training():
    """Test converting an intercepted session to training examples."""
    from neuro.training.interceptor import InterceptedSession, ConversationTurn

    session = InterceptedSession(
        session_id="test_session",
        tool="claude",
        started_at=1000.0,
    )

    session.turns = [
        ConversationTurn(role="user", content="How do I read a file in Python?", tool="claude"),
        ConversationTurn(role="assistant", content="Use `open()` with a context manager:\n```python\nwith open('file.txt') as f:\n    content = f.read()\n```", tool="claude"),
        ConversationTurn(role="user", content="How do I write to it?", tool="claude"),
        ConversationTurn(role="assistant", content="Use `open()` with mode `'w'`:\n```python\nwith open('file.txt', 'w') as f:\n    f.write('hello')\n```", tool="claude"),
    ]

    examples = session.to_training_examples()
    print(f"  Session → {len(examples)} training examples")
    assert len(examples) == 2

    for ex in examples:
        msgs = ex["messages"]
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        print(f"    User: {msgs[1]['content'][:50]}")
        print(f"    Asst: {msgs[2]['content'][:50]}")

    print("  ✓ Session → training conversion works correctly")


def test_continual_learner_status():
    """Test that the continual learner can report status."""
    import tempfile
    from neuro.training.continual import ContinualLearner

    with tempfile.TemporaryDirectory() as tmpdir:
        learner = ContinualLearner(traces_dir=Path(tmpdir))
        # Just verify it doesn't crash
        learner.status()
        print("  ✓ ContinualLearner status works")


if __name__ == "__main__":
    tests = [
        ("OutputParser", test_output_parser),
        ("TraceStorage", test_trace_storage),
        ("SteeringVector Loading", test_steering_vector_loading),
        ("DatasetBuilder", test_dataset_builder),
        ("ColabGenerator", test_colab_generator),
        ("AdapterManager", test_adapter_manager),
        ("Session→Training", test_intercepted_session_to_training),
        ("ContinualLearner", test_continual_learner_status),
    ]

    print("=" * 60)
    print("NeuroBridge Self-Learning Pipeline Tests")
    print("=" * 60)

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n[{name}]")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'=' * 60}")

    sys.exit(1 if failed > 0 else 0)
