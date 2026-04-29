"""
Chat CLI for the 3-Phase Autonomous Web Agent.

Phase 1: Receives URL, triggers page preparation
Phase 2: After analysis, user picks Action (Pydoll) or Chat (Node RAG on same job)
Phase 3: Runs with fixed mode or auto intent when mode not set
"""

import asyncio
import os
import re
import sys
from typing import Optional

# URL extraction helper
def extract_url(text: str) -> Optional[str]:
    """Extract first http(s) URL from text."""
    match = re.search(r'https?://[^\s<>"\'\)]+', text)
    return match.group(0) if match else None


def _job_folder_label(job_dir: Optional[str]) -> str:
    if not job_dir:
        return "(unknown)"
    return os.path.basename(os.path.normpath(job_dir))


def prompt_mode_choice() -> str:
    """Return 'action' or 'chat' after user picks 1 or 2."""
    print()
    print("  Choose how to continue (same job / saved nodes):")
    print("    [1] Actions — automate the page with Pydoll (plan → generated steps)")
    print("    [2] Chat    — ask about this page (semantic search over indexed DOM nodes)")
    print("  Tip: type  menu  or  m  later to change mode.")
    print()
    while True:
        raw = input("Mode [1/2]> ").strip().lower()
        if raw in ("1", "action", "a"):
            return "action"
        if raw in ("2", "chat", "c"):
            return "chat"
        print("  Please enter 1 (Actions) or 2 (Chat).")


async def run_chat_cli():
    """Main Chat CLI REPL for the 3-phase autonomous web agent."""
    from .pipelines.three_phase_pipeline import run_three_phase_agent

    print("=" * 60)
    print("  Autonomous Web Agent — 3-Phase Chat CLI")
    print("=" * 60)
    print()
    print("Commands:")
    print("  <URL>           — Load and prepare a page (Phase 1)")
    print("  menu / m        — After page is ready: pick Actions vs Chat again")
    print("  exit / quit     — Exit the agent")
    print()

    page_ready = False
    agent_state = None
    session_mode: Optional[str] = None  # "action" | "chat"

    while True:
        try:
            if not page_ready:
                prompt = "URL> "
            elif session_mode == "action":
                prompt = "Action> "
            elif session_mode == "chat":
                prompt = "Chat> "
            else:
                prompt = "You> "

            user_input = input(prompt).strip()
            if not user_input:
                continue

            low = user_input.lower()
            if low in ("exit", "quit"):
                print("Goodbye.")
                break

            if not page_ready:
                # Phase 1: URL received — trigger page preparation
                url = extract_url(user_input)
                if not url:
                    print("No URL found. Please enter a valid http(s) URL.")
                    continue

                print(f"\n[Phase 1] Preparing page: {url}")
                print("  Fetching HTML, handling blockers, expanding content, building DOM graph...")
                try:
                    agent_state = await run_three_phase_agent(url=url, user_demand=None)
                    if agent_state and agent_state.get("status") == "ready_for_demand":
                        page_ready = True
                        jd = agent_state.get("job_dir") or ""
                        print("\n" + "=" * 60)
                        print("  Page analyzed and ready.")
                        print(f"  Job id: {_job_folder_label(jd)}")
                        print("=" * 60)
                        session_mode = prompt_mode_choice()
                        if session_mode == "action":
                            print("\n  Action mode — describe what to do on the page (Pydoll).")
                        else:
                            print("\n  Chat mode — questions use the indexed nodes for this job only.")
                    else:
                        err = agent_state.get("error", "Unknown error") if agent_state else "Phase 1 failed"
                        print(f"\n[ERROR] {err}")
                except Exception as e:
                    print(f"\n[ERROR] {e}")
                    import traceback
                    traceback.print_exc()

            else:
                if low in ("menu", "m", "mode"):
                    session_mode = prompt_mode_choice()
                    if session_mode == "action":
                        print("\n  Switched to Action mode.")
                    else:
                        print("\n  Switched to Chat mode.")
                    continue

                user_demand = user_input
                mode_label = "Actions (Pydoll)" if session_mode == "action" else "Chat (Node RAG)"
                print(f"\n[Phase 3] {mode_label}: {user_demand}")
                try:
                    result_state = await run_three_phase_agent(
                        url=agent_state.get("url"),
                        user_demand=user_demand,
                        prepared_state=agent_state,
                        phase3_mode=session_mode,
                    )
                    result = result_state.get("result", "")
                    if result:
                        print("\n" + "-" * 40)
                        print("Agent:", result)
                        print("-" * 40)
                    else:
                        print("\n[INFO] No result returned.")
                except Exception as e:
                    print(f"\n[ERROR] {e}")
                    import traceback
                    traceback.print_exc()

        except EOFError:
            print("\nGoodbye.")
            break
        except KeyboardInterrupt:
            print("\nInterrupted. Type 'exit' to quit.")
            continue


def main():
    """Entry point for the Chat CLI."""
    asyncio.run(run_chat_cli())


if __name__ == "__main__":
    main()
