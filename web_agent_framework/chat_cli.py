"""
Chat CLI for the 3-Phase Autonomous Web Agent.

Phase 1: Receives URL, triggers page preparation
Phase 2: Signals readiness, listens for user demand
Phase 3: Executes demand, returns result via CLI
"""

import asyncio
import re
import sys
from typing import Optional

# URL extraction helper
def extract_url(text: str) -> Optional[str]:
    """Extract first http(s) URL from text."""
    match = re.search(r'https?://[^\s<>"\'\)]+', text)
    return match.group(0) if match else None


async def run_chat_cli():
    """Main Chat CLI REPL for the 3-phase autonomous web agent."""
    from .pipelines.three_phase_pipeline import run_three_phase_agent

    print("=" * 60)
    print("  Autonomous Web Agent — 3-Phase Chat CLI")
    print("=" * 60)
    print()
    print("Commands:")
    print("  <URL>           — Load and prepare a page (Phase 1)")
    print("  <demand>        — After page is ready: ask a question or give a command")
    print("  exit / quit     — Exit the agent")
    print()

    page_ready = False
    agent_state = None

    while True:
        try:
            if page_ready:
                prompt = "You> "
            else:
                prompt = "URL> "

            user_input = input(prompt).strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
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
                        print("\n" + "=" * 60)
                        print("  Page analyzed and ready.")
                        print("  What would you like me to do?")
                        print("=" * 60)
                    else:
                        err = agent_state.get("error", "Unknown error") if agent_state else "Phase 1 failed"
                        print(f"\n[ERROR] {err}")
                except Exception as e:
                    print(f"\n[ERROR] {e}")
                    import traceback
                    traceback.print_exc()

            else:
                # Phase 2 → Phase 3: User demand — execute and return result
                user_demand = user_input
                print(f"\n[Phase 3] Executing: {user_demand}")
                try:
                    result_state = await run_three_phase_agent(
                        url=agent_state.get("url"),
                        user_demand=user_demand,
                        prepared_state=agent_state,
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
