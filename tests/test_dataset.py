import kogitune
import kogitune.adhocs as adhoc
import pytest

def test_stream_csv():
    url = 'https://raw.githubusercontent.com/kkuramitsu/kogitune/refs/heads/main/test_data/random_bmi.csv'
    stream = kogitune.load('datastream', url, names=['height', 'weight', 'label'], skiprows=1)
    assert sum(1 for _ in stream.samples()) == 500
