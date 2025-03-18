import subprocess


def test_library_mode_extract_pdf():
    process = subprocess.run(["python", "examples/library_mode_example.py"], capture_output=True, text=True)

    try:
        assert process.returncode == 0
        assert "Chasing a squirrel" in process.stdout
    except:
        print(process.stdout)
        print(process.stderr)
        raise
