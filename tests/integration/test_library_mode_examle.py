import subprocess


def test_library_mode_extract_pdf():
    process = subprocess.run(["python", "examples/library_mode_example.py"], capture_output=True, text=True)

    try:
        assert process.returncode == 0
        # pdfium text
        assert "A sample document with headings and placeholder text" in process.stdout
        # table in markdown
        assert "| Dog | Chasing a squirrel | In the front yard |" in process.stdout
        # chart labels
        assert "Tweeter - Midrange - Midwoofer - Subwoofer" in process.stdout
    except:
        print(process.stdout)
        print(process.stderr)
        raise
