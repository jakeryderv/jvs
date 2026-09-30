from cyclopts import App

app = App()


@app.command
def hello(name: str = "world") -> None:
    print(f"Hello, {name}!")


@app.command
def tui() -> None:
    from jvs._tui.app import main as run_tui

    run_tui()


@app.command
def repl() -> None:
    from jvs._repl.app import main as run_repl

    run_repl()


def main() -> None:
    app()
