# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Regression coverage for the standalone packaging boundary."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

from click.testing import CliRunner
import numpy as np
import pytest
import yaml

from synthpdf.cli import cli
from synthpdf import document_generator as generator


def test_cli_has_only_unconditional_generation():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert list(cli.commands) == ["generate"]
    for args in [["generate", "--topics-file", "topic.yaml"], ["generate", "--config", "config.yaml"]]:
        result = CliRunner().invoke(cli, args)
        assert result.exit_code != 0
        assert "No such option" in result.output


def test_generation_config_is_portable_and_does_not_store_credentials(tmp_path, monkeypatch):
    import synthpdf._pipeline as pipeline

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "test-model")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-credential-never-serialize")
    run = Mock()
    monkeypatch.setattr(pipeline, "run_config", run)
    result = generator.generate_documents(num_pages=4, num_pdfs=1, seed=42)
    config = yaml.safe_load(result["config_path"].read_text(encoding="utf-8"))
    assert run.call_count == 1
    assert config["azure_openai"]["deployment"] == "test-model"
    assert "test-credential-never-serialize" not in result["config_path"].read_text()
    for path in [config["topics_file"], config["template_engine"]["templates_dir"],
                 config["rendering"]["branding"]["logos_dir"]]:
        assert Path(path).exists()
    assert result["output_dir"].is_relative_to(tmp_path)


def test_missing_model_settings_fail_before_writing_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
        generator.generate_documents(num_pdfs=1, seed=42)
    assert not (tmp_path / "output").exists()


def test_packaged_templates_fonts_and_handwriting():
    from synthpdf.shared.handwriting import simulate_handwriting
    from synthpdf.template_engine.index import TemplateIndex
    from synthpdf.variants.forms import _FILL_FONTS_DIR

    resources = generator._RESOURCES
    index = TemplateIndex(resources / "jinja2_templates" / "specs",
                          resources / "jinja2_templates" / "templates")
    assert len(index) == 107
    assert len(list((resources / "placeholders").glob("*.png"))) == 183
    assert len(list(_FILL_FONTS_DIR.glob("*.ttf"))) == 10
    first = np.asarray(simulate_handwriting(["Synthetic sample"], seed=42))
    second = np.asarray(simulate_handwriting(["Synthetic sample"], seed=42))
    assert np.array_equal(first, second)
    assert first.std() > 0


def test_llm_generation_still_uses_client_and_existing_qa_helper():
    from synthpdf.llm.content_generator import ContentGenerator, TopicSpec

    client = Mock()
    client.complete.return_value = json.dumps({"qa": [{
        "question": 'In the "Example" section, what is the amount?',
        "answer": "100",
        "assertions": [{"text": "The response must state 100."}],
    }]})
    content = ContentGenerator(client, TopicSpec("sample", "Synthetic example"))
    result = content.generate_qa_candidates("**Example**\n\nThe amount is 100.", n=1)
    assert result[0]["answer"] == "100"
    assert client.complete.call_args.kwargs["tier"] == "medium"


def test_mixed_pipeline_smoke_with_replayed_generation(tmp_path, monkeypatch):
    from synthpdf import _pipeline as pipeline
    from synthpdf.llm.client import AzureOpenAIClient
    from synthpdf.llm.content_generator import ContentGenerator
    from synthpdf.types import Variant

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "test-model")
    config = generator._assemble_config(4, 1, tmp_path, 42)
    config["recipes"][0]["layout_probabilities"] = {"text": 1.0}
    config["rendering"]["backend"] = "playwright"
    path = tmp_path / "internal_config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    qa = {"question": 'In "Packaging Example", what is the amount?',
          "answer": "100", "assertions": [{"text": "The response must state 100."}]}
    variant = Variant(
        "text_sample", "text", {"heading": "Packaging Example"},
        {"markdown": "**Packaging Example**\n\nThe amount is 100."},
        {"text_content": "The amount is 100.", "qa_candidates": [qa]},
    )
    pool = {"texts": [variant], "tables": [], "images": [], "annotations": [],
            "forms": [], "scanned": [], "infographics": []}
    monkeypatch.setattr(AzureOpenAIClient, "__init__", lambda self, cfg: None)
    generate_pool = Mock(return_value=pool)
    monkeypatch.setattr(pipeline, "_generate_llm_variant_pool", generate_pool)
    monkeypatch.setattr(ContentGenerator, "generate_document_title", lambda self: "Packaging Example")
    monkeypatch.setattr(ContentGenerator, "generate_document_subtitle", lambda self: "Synthetic sample")
    pipeline.run_config(path)
    assert generate_pool.call_count == 1
    manifests = list((tmp_path / "manifests").glob("*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    pdf = Path(manifest["files"]["pdf"])
    assert pdf.read_bytes().startswith(b"%PDF-")
    assert manifest["qa_candidates"][0]["answer"] == "100"
    for extension in [".xlsx", ".json", ".tsv", ".yml"]:
        assert (tmp_path / "quest" / (manifest["doc_id"] + extension)).exists()
