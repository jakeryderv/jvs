from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Static


class JVSApp(App):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("JVS TUI")
        yield Footer()


def main() -> None:
    JVSApp().run()
