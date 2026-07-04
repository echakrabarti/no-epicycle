from solution import reverse_string

def test_basic():
    assert reverse_string("hello") == "olleh"

def test_empty():
    assert reverse_string("") == ""

def test_single():
    assert reverse_string("a") == "a"

def test_spaces():
    assert reverse_string("hello world") == "dlrow olleh"

def test_numbers():
    assert reverse_string("12345") == "54321"