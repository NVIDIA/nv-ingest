import subprocess


def test_library_mode_extract_pdf():
    result = subprocess.run(["python", "examples/library_mode_example.py"], capture_output=True, text=True)

    try:
        assert "Chasing a squirrel" in result.stdout
    except:
        print("Standard Output:")
        print(result.stdout)
        print("Standard Error:")
        print(result.stderr)
        raise
