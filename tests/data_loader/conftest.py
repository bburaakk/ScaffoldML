from claims_ml.src.data_loader import DataLoader
import pytest

@pytest.fixture
def data_loader():
    return DataLoader()

