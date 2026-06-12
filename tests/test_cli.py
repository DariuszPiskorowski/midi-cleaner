import json

from typer.testing import CliRunner

from midi_cleaner import __version__
from midi_cleaner.cli import app


runner = CliRunner()


def test_cli_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_doctor_json_is_parseable() -> None:
    result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code in {0, 1}
    payload = json.loads(result.stdout)
    assert payload["project_name"] == "midi-cleaner"
    assert payload["python"]["required_major_minor"] == "3.11"
    assert "external_tools" in payload
