"""Sizing a submitted run and leaving out report.html — no JVM needed."""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

import gnomepy.reporting
from gnomepy import cli, remote

CONFIG = "strategy:\n  class_name: some.module:Strategy\nlistings:\n  - listing_id: 1\n"


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG)
    return str(path)


@pytest.fixture
def sent(monkeypatch):
    """The bodies the controller would receive."""
    bodies = []

    def request(method, path, json=None, **_):
        bodies.append(json)
        return {"run_id": "r1", "job_count": 1, "status": "SUBMITTED", "batch_job_id": "b1"}

    monkeypatch.setattr(remote, "_request", request)
    return bodies


def submit(config_file, *args):
    result = CliRunner().invoke(cli.main, ["backtest", "submit", "--config", config_file, *args])
    assert result.exit_code == 0, result.output
    return result


def test_a_submit_sends_its_size_and_leaves_out_the_report(config_file, sent):
    submit(config_file, "--cpu", "2", "--memory-gib", "16", "--no-report")
    assert sent == [{"config": CONFIG, "research_commit": "main", "cpu": 2, "memory_gib": 16, "report": False}]


def test_a_submit_without_options_keeps_the_job_definitions_size_and_the_report(config_file, sent):
    submit(config_file)
    assert sent == [{"config": CONFIG, "research_commit": "main"}]


class _Report:
    """BacktestReport writing files the way the real one does, and failing on the HTML when told to."""

    def __init__(self, results, fail_html=False):
        self.fail_html = fail_html

    def save_summary(self, path):
        with open(path, "w") as f:
            json.dump({"final_pnl": 1.0}, f)

    def save_html(self, path):
        if self.fail_html:
            raise MemoryError("out of memory building the report")
        with open(path, "w") as f:
            f.write("<html></html>")


@pytest.mark.parametrize("html, files", [(True, {"summary.json", "report.html"}), (False, {"summary.json"})])
def test_the_summary_is_written_with_or_without_the_html(tmp_path, monkeypatch, html, files):
    monkeypatch.setattr(gnomepy.reporting, "BacktestReport", _Report)
    cli._generate_report(object(), str(tmp_path), html)
    assert {p.name for p in tmp_path.iterdir()} == files


def test_a_report_that_fails_still_leaves_the_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(gnomepy.reporting, "BacktestReport", lambda results: _Report(results, fail_html=True))
    cli._generate_report(object(), str(tmp_path))
    assert {p.name for p in tmp_path.iterdir()} == {"summary.json"}
