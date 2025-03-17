import subprocess


def test_library_mode_extract_pdf():
    result = subprocess.run(["python", "examples/library_mode_example.py"], capture_output=True, text=True)

    assert "Chasing a squirrel" in result.stdout
