from jvs._cli.app import hello


def test_hello(capsys):
    hello("Jake")
    assert capsys.readouterr().out == "Hello, Jake!\n"
