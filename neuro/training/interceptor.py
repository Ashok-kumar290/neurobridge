"""Terminal Interceptor — captures expert AI sessions as training data.

Wraps any CLI tool (claude, codex, aider, etc.) in a PTY and silently
records the full (user → assistant) conversation as training pairs.

This is the core data collection mechanism for NeuroBridge's self-learning loop.
The captured traces become fine-tuning data for the local Super-Qwen model.

Usage:
    # Wrap a claude code session
    interceptor = TerminalInterceptor(tool="claude", storage_dir="/path/to/traces")
    interceptor.run()

    # Or via CLI
    python -m neuro.training.interceptor --tool claude
"""

from __future__ import annotations

import json
import os
import pty
import re
import select
import signal
import struct
import sys
import termios
import time
import fcntl
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ConversationTurn:
    """A single turn in an intercepted conversation."""
    role: str           # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    tool: str = ""
    
    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "tool": self.tool,
        }


@dataclass 
class InterceptedSession:
    """A complete intercepted session with metadata."""
    session_id: str
    tool: str
    started_at: float
    turns: list[ConversationTurn] = field(default_factory=list)
    raw_output: str = ""
    finished_at: float = 0.0
    
    def to_training_examples(self) -> list[dict]:
        """Convert session turns into ChatML training examples."""
        examples = []
        
        # Group into (user, assistant) pairs
        i = 0
        while i < len(self.turns) - 1:
            if self.turns[i].role == "user" and self.turns[i + 1].role == "assistant":
                user_msg = self.turns[i].content.strip()
                asst_msg = self.turns[i + 1].content.strip()
                
                # Skip trivial exchanges
                if len(user_msg) < 5 or len(asst_msg) < 10:
                    i += 1
                    continue
                
                examples.append({
                    "messages": [
                        {"role": "system", "content": "You are NeuroBridge, a local AI coding assistant. Provide accurate, concise responses."},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": asst_msg},
                    ],
                    "metadata": {
                        "session_id": self.session_id,
                        "tool": self.tool,
                        "timestamp": self.turns[i].timestamp,
                    }
                })
                i += 2
            else:
                i += 1
        
        return examples
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "tool": self.tool,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "turns": [t.to_dict() for t in self.turns],
            "num_examples": len(self.to_training_examples()),
        }


class OutputParser:
    """Parses raw terminal output into conversation turns.
    
    Different tools have different output formats:
    - claude: Uses markdown-style output with clear user/assistant boundaries
    - codex: Similar but with different delimiters
    - aider: Has a specific prompt format
    
    This parser uses heuristics to detect turn boundaries.
    """
    
    # Common patterns that indicate a user input prompt
    USER_PROMPT_PATTERNS = [
        r'^\s*>\s',                    # > prompt
        r'^\s*\$\s',                   # $ prompt
        r'^\s*claude>\s',              # claude> prompt
        r'^\s*aider>\s',              # aider> prompt
        r'^\s*codex>\s',              # codex> prompt
        r'^\s*human>\s',              # human> prompt
        r'^\s*you:\s',                # you: prompt
        r'^\s*\[\d+\]\s*>\s',        # [1] > numbered prompt
        r'^❯\s',                      # fancy prompt
        r'^→\s',                      # arrow prompt
    ]
    
    # Patterns that indicate assistant output start
    ASSISTANT_PATTERNS = [
        r'^\s*claude:\s',
        r'^\s*assistant:\s',
        r'^\s*codex:\s',
        r'^\s*AI:\s',
    ]
    
    def __init__(self, tool: str = "claude"):
        self.tool = tool
        self.buffer = ""
        self.turns: list[ConversationTurn] = []
        self.current_role = "assistant"  # Most tools start with assistant output
        self.current_content = ""
        
    def feed(self, data: str) -> list[ConversationTurn]:
        """Feed raw terminal output and extract any complete turns."""
        self.buffer += data
        new_turns = []
        
        # Process line by line
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = self._strip_ansi(line)
            
            # Detect role switches
            if self._is_user_prompt(line):
                # Save previous content (whatever role it was)
                if self.current_content.strip():
                    turn = ConversationTurn(
                        role=self.current_role,
                        content=self.current_content.strip(),
                        tool=self.tool,
                    )
                    self.turns.append(turn)
                    new_turns.append(turn)
                
                self.current_role = "user"
                # Extract user input from the prompt line
                user_input = self._extract_user_input(line)
                self.current_content = user_input + "\n" if user_input else ""
                    
            elif self.current_role == "user" and not self._is_user_prompt(line) and not line.strip() == "":
                # Non-prompt line after user input → assistant is responding
                # Save user turn first
                if self.current_content.strip():
                    turn = ConversationTurn(
                        role="user",
                        content=self.current_content.strip(),
                        tool=self.tool,
                    )
                    self.turns.append(turn)
                    new_turns.append(turn)
                
                self.current_role = "assistant"
                self.current_content = line + "\n"
            else:
                self.current_content += line + "\n"
        
        return new_turns
    
    def flush(self) -> Optional[ConversationTurn]:
        """Flush any remaining content as a final turn."""
        if self.current_content.strip():
            turn = ConversationTurn(
                role=self.current_role,
                content=self.current_content.strip(),
                tool=self.tool,
            )
            self.turns.append(turn)
            self.current_content = ""
            return turn
        return None
    
    def _is_user_prompt(self, line: str) -> bool:
        """Check if a line looks like a user input prompt."""
        for pattern in self.USER_PROMPT_PATTERNS:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        return False
    
    def _is_assistant_start(self, line: str) -> bool:
        """Check if a line looks like the start of assistant output."""
        for pattern in self.ASSISTANT_PATTERNS:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        return False
    
    def _extract_user_input(self, line: str) -> str:
        """Extract the user's input from a prompt line."""
        # Remove common prompt prefixes
        for pattern in self.USER_PROMPT_PATTERNS:
            line = re.sub(pattern, '', line, flags=re.IGNORECASE)
        return line.strip()
    
    @staticmethod
    def _strip_ansi(text: str) -> str:
        """Remove ANSI escape sequences from text."""
        return re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)


class TraceStorage:
    """Append-only JSONL storage for intercepted sessions on HDD.
    
    Uses append-only writes so data is never lost, even if the process
    crashes mid-session. Each line is a complete JSON object.
    """
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Main trace files
        self.buffer_path = self.storage_dir / "replay_buffer.jsonl"
        self.sessions_dir = self.storage_dir / "sessions"
        self.sessions_dir.mkdir(exist_ok=True)
    
    def append_example(self, example: dict) -> None:
        """Append a single training example to the replay buffer."""
        with open(self.buffer_path, "a") as f:
            f.write(json.dumps(example) + "\n")
    
    def save_session(self, session: InterceptedSession) -> Path:
        """Save a complete session to its own file."""
        session_path = self.sessions_dir / f"{session.session_id}.json"
        with open(session_path, "w") as f:
            json.dump(session.to_dict(), f, indent=2)
        return session_path
    
    def get_buffer_stats(self) -> dict:
        """Get stats about the replay buffer."""
        if not self.buffer_path.exists():
            return {"total_examples": 0, "size_bytes": 0}
        
        count = 0
        with open(self.buffer_path) as f:
            for _ in f:
                count += 1
        
        return {
            "total_examples": count,
            "size_bytes": self.buffer_path.stat().st_size,
            "size_mb": self.buffer_path.stat().st_size / (1024 * 1024),
        }
    
    def tail_examples(self, n: int = 10) -> list[dict]:
        """Read the last N examples from the buffer."""
        if not self.buffer_path.exists():
            return []
        
        lines = []
        with open(self.buffer_path) as f:
            for line in f:
                lines.append(line)
                if len(lines) > n:
                    lines.pop(0)
        
        examples = []
        for line in lines:
            try:
                examples.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
        return examples


class TerminalInterceptor:
    """Wraps a CLI tool in a PTY and captures the conversation.
    
    The user sees and interacts with the tool normally.
    In the background, all I/O is captured and parsed into
    training examples that are stored to HDD.
    """
    
    def __init__(
        self,
        tool: str = "claude",
        tool_args: list[str] | None = None,
        storage_dir: Path | str = "/media/seyominaoto/x/neurobridge/traces",
    ):
        self.tool = tool
        self.tool_args = tool_args or []
        self.storage = TraceStorage(Path(storage_dir))
        self.parser = OutputParser(tool=tool)
        self.session: Optional[InterceptedSession] = None
        self._child_pid: Optional[int] = None
        self._master_fd: Optional[int] = None
        
    def run(self) -> InterceptedSession:
        """Run the tool in a PTY and capture the session."""
        session_id = f"intercept_{self.tool}_{int(time.time())}"
        self.session = InterceptedSession(
            session_id=session_id,
            tool=self.tool,
            started_at=time.time(),
        )
        
        print(f"[neuro] Interceptor active — capturing {self.tool} session")
        print(f"[neuro] Session: {session_id}")
        print(f"[neuro] Storage: {self.storage.storage_dir}")
        print(f"[neuro] Use the tool normally. All exchanges are being captured.")
        print(f"[neuro] {'='*60}\n")
        
        # Save original terminal settings
        old_tty = termios.tcgetattr(sys.stdin)
        
        try:
            # Fork a child process with a PTY
            self._child_pid, self._master_fd = pty.fork()
            
            if self._child_pid == 0:
                # Child process — exec the tool
                cmd = [self.tool] + self.tool_args
                os.execvp(cmd[0], cmd)
            else:
                # Parent process — proxy I/O and capture
                self._proxy_io(old_tty)
        except Exception as e:
            print(f"\n[neuro] Interceptor error: {e}")
        finally:
            # Restore terminal
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)
            
            # Flush remaining parser content
            final_turn = self.parser.flush()
            
            # Finalize session
            self.session.finished_at = time.time()
            self.session.turns = self.parser.turns
            
            # Save session and extract training examples
            self._save_results()
            
        return self.session
    
    def _proxy_io(self, old_tty) -> None:
        """Proxy I/O between user terminal and child PTY."""
        # Set raw mode on stdin so keystrokes pass through
        import tty as tty_module
        tty_module.setraw(sys.stdin.fileno())
        
        # Match terminal size
        self._sync_window_size()
        
        # Handle window resize
        def handle_winch(signum, frame):
            self._sync_window_size()
        signal.signal(signal.SIGWINCH, handle_winch)
        
        try:
            while True:
                rlist, _, _ = select.select(
                    [sys.stdin, self._master_fd], [], [], 0.1
                )
                
                if sys.stdin in rlist:
                    # User typed something — forward to child
                    data = os.read(sys.stdin.fileno(), 1024)
                    if not data:
                        break
                    os.write(self._master_fd, data)
                
                if self._master_fd in rlist:
                    # Child produced output — show to user AND capture
                    try:
                        data = os.read(self._master_fd, 4096)
                    except OSError:
                        break
                    if not data:
                        break
                    
                    # Show to user
                    os.write(sys.stdout.fileno(), data)
                    
                    # Capture for training
                    try:
                        text = data.decode('utf-8', errors='replace')
                        self.session.raw_output += text
                        new_turns = self.parser.feed(text)
                        
                        # Save turns as they come in (crash-safe)
                        for turn in new_turns:
                            if turn.role == "assistant":
                                # Check if we have a preceding user turn to pair with
                                idx = self.parser.turns.index(turn)
                                if idx > 0 and self.parser.turns[idx - 1].role == "user":
                                    user_turn = self.parser.turns[idx - 1]
                                    example = {
                                        "messages": [
                                            {"role": "system", "content": "You are NeuroBridge, a local AI coding assistant."},
                                            {"role": "user", "content": user_turn.content},
                                            {"role": "assistant", "content": turn.content},
                                        ],
                                        "metadata": {
                                            "session_id": self.session.session_id,
                                            "tool": self.tool,
                                            "timestamp": turn.timestamp,
                                        }
                                    }
                                    self.storage.append_example(example)
                    except Exception:
                        pass  # Never crash the interceptor
                        
        except (IOError, OSError):
            pass
    
    def _sync_window_size(self) -> None:
        """Sync terminal window size to the child PTY."""
        try:
            win = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, b'\x00' * 8)
            fcntl.ioctl(self._master_fd, termios.TIOCSWINSZ, win)
        except Exception:
            pass
    
    def _save_results(self) -> None:
        """Save session results and print summary."""
        if not self.session:
            return
            
        # Save full session
        session_path = self.storage.save_session(self.session)
        
        # Extract and save training examples
        examples = self.session.to_training_examples()
        for ex in examples:
            self.storage.append_example(ex)
        
        # Print summary
        duration = self.session.finished_at - self.session.started_at
        stats = self.storage.get_buffer_stats()
        
        print(f"\n{'='*60}")
        print(f"[neuro] Session captured: {self.session.session_id}")
        print(f"[neuro] Duration: {duration:.0f}s")
        print(f"[neuro] Turns captured: {len(self.session.turns)}")
        print(f"[neuro] Training examples: {len(examples)}")
        print(f"[neuro] Session saved: {session_path}")
        print(f"[neuro] Total buffer: {stats['total_examples']} examples ({stats.get('size_mb', 0):.1f} MB)")
        print(f"{'='*60}")


def main():
    """CLI entry point for the interceptor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuroBridge Terminal Interceptor")
    parser.add_argument("--tool", default="claude", help="CLI tool to wrap (default: claude)")
    parser.add_argument("--args", nargs="*", default=[], help="Arguments to pass to the tool")
    parser.add_argument("--storage", default="/media/seyominaoto/x/neurobridge/traces",
                       help="Storage directory for traces")
    parser.add_argument("--stats", action="store_true", help="Show buffer stats and exit")
    
    args = parser.parse_args()
    
    if args.stats:
        storage = TraceStorage(Path(args.storage))
        stats = storage.get_buffer_stats()
        print(f"Replay buffer: {stats['total_examples']} examples ({stats.get('size_mb', 0):.1f} MB)")
        recent = storage.tail_examples(3)
        if recent:
            print(f"\nMost recent examples:")
            for ex in recent:
                msgs = ex.get("messages", [])
                if len(msgs) >= 3:
                    user = msgs[1]["content"][:80]
                    asst = msgs[2]["content"][:80]
                    print(f"  User: {user}...")
                    print(f"  Asst: {asst}...")
                    print()
        return
    
    interceptor = TerminalInterceptor(
        tool=args.tool,
        tool_args=args.args,
        storage_dir=Path(args.storage),
    )
    interceptor.run()


if __name__ == "__main__":
    main()
