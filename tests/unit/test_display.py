from rich import print

from lapidarist.display import header


def test_header():
    print(header())
    assert True, "Printed header"
