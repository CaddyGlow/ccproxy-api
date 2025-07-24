from textual.app import App, ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import DataTable, Static


# The data for the permission request
REQUEST_DATA = {
    "Tool": "TestTool",
    "Request ID": "test-123...",
    "action": "create_file",
    "file_path": "/tmp/test_file.txt",
    "content": "This is a test file content for demon...",
    "permissions": "0644",
    "description": "Testing the new synchronous Rich impl...",
}


class PermissionDialog(ModalScreen[bool]):
    """A modal screen for permission requests."""

    timeout = reactive(30)

    # FIX: The CSS has been updated to use 'center middle' for content-align
    CSS = """
    PermissionDialog {
        align: center middle;
    }

    #dialog {
        width: 80;
        height: auto;
        padding: 1 2;
        border: thick blue 80%;
        background: $surface;
    }

    #title {
        content-align: center middle; /* FIX: Was 'center' */
        width: 100%;
        margin-bottom: 1;
    }

    DataTable {
        height: auto;
        margin-bottom: 1;
    }

    #timeout-label {
        content-align: center middle; /* FIX: Was 'center' */
        width: 100%;
        text-style: bold;
        color: $warning;
    }

    #instructions {
        content-align: center middle; /* FIX: Was 'center' */
        width: 100%;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Static("Permission Request", id="title")
            yield DataTable()
            yield Static(id="timeout-label")
            yield Static("Type 'y' to allow, anything else to deny", id="instructions")

    def on_mount(self) -> None:
        """Set up the dialog when it's mounted."""
        table = self.query_one(DataTable)
        table.add_column("Parameter", width=15)
        table.add_column("Value")
        for key, value in REQUEST_DATA.items():
            table.add_row(f"  {key}", value)

        self.update_timeout_label()
        self.timer = self.set_interval(1.0, self.countdown)

    def countdown(self) -> None:
        """Called every second by the timer."""
        self.timeout -= 1
        if self.timeout <= 0:
            self.timer.stop()
            self.dismiss(False)

    def watch_timeout(self, timeout: int) -> None:
        """Reactive method called when self.timeout changes."""
        self.update_timeout_label()

    def update_timeout_label(self) -> None:
        """Updates the text of the timeout label."""
        self.query_one("#timeout-label").update(f"Timeout in {self.timeout}s")

    def on_key(self, event) -> None:
        """Handle user key presses."""
        self.timer.stop()
        if event.key == "y":
            self.dismiss(True)
        else:
            self.dismiss(False)


class PermissionApp(App):
    """A Textual app to show a permission dialog."""

    def on_mount(self) -> None:
        """Mount the app and show the dialog."""

        def check_result(allowed: bool) -> None:
            """Callback to handle the dialog's result."""
            if allowed:
                self.exit("Permission ALLOWED ✅")
            else:
                self.exit("Permission DENIED ❌")

        self.push_screen(PermissionDialog(), check_result)


if __name__ == "__main__":
    app = PermissionApp()
    result = app.run()
    if result:
        print(result)
