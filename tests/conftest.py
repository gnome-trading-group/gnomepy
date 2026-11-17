import pytest

import gnomepy.reporting.report as report_module


class _NoSettlements:
    """A registry that knows no listings, so a report values every held position at its last mark."""

    def get_listing(self, **_):
        return []

    def get_event_contracts(self, **_):
        return []


@pytest.fixture(autouse=True)
def _reports_read_no_registry(monkeypatch):
    # A report looks up how held listings settled; tests must never reach the real registry for it.
    monkeypatch.setattr(report_module, "RegistryClient", _NoSettlements)
