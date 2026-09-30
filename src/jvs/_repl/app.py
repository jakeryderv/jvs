from prompt_toolkit import PromptSession


def main() -> None:
    session = PromptSession()

    while True:
        try:
            command = session.prompt("jvs>").strip()

            if command in {"exit", "quit"}:
                break

            if command:
                print(f"Command: {command}")

        except (EOFError, KeyboardInterrupt):
            break
