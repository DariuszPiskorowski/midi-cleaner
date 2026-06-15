from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os

from pydantic import ValidationError

from midi_cleaner.ai_completion.compact_pack import build_ai_request_pack
from midi_cleaner.ai_completion.export import (
    AICompletionValidationResult,
    export_ai_completion_midi,
    validate_ai_completion_notes,
)
from midi_cleaner.ai_completion.models import (
    AIPatternCompletionNote,
    AIPatternCompletionOutput,
    AIPatternCompletionReport,
)
from midi_cleaner.ai_completion.openai_client import (
    OpenAIPatternCompletionClient,
    OpenAIPatternCompletionClientError,
)
from midi_cleaner.ai_completion.pattern_pack import PatternPackBuildError, build_pattern_pack
from midi_cleaner.ai_completion.prompt import build_ai_completion_prompts


class AIPatternCompletionError(Exception):
    """Raised when ai_pattern_completion cannot run successfully."""


@dataclass(frozen=True)
class AIPatternCompletionParameters:
    layer: str = "bass"
    model: str | None = None
    output_dir: Path | None = None
    dry_run: bool = False
    max_completion_notes: int = 64
    temperature: float = 0.2
    keep_ai_json: bool = True


_MAX_AI_PROMPT_CHARS = 250_000


def complete_ai_pattern_completion(
    project_dir: Path,
    params: AIPatternCompletionParameters,
    ai_client: OpenAIPatternCompletionClient | None = None,
) -> AIPatternCompletionReport:
    project_dir = project_dir.resolve()
    analysis_output_dir = project_dir / "analysis" / "ai_pattern_completion"
    midi_output_dir = _resolve_output_dir(project_dir, params.output_dir)
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    midi_output_dir.mkdir(parents=True, exist_ok=True)

    pattern_pack_path = analysis_output_dir / "pattern_pack.json"
    ai_request_pack_path = analysis_output_dir / "ai_request_pack.json"
    prompt_path = analysis_output_dir / "ai_prompt.txt"
    ai_json_path = analysis_output_dir / "bass_ai_completion.json"
    report_path = analysis_output_dir / "bass_ai_completion_report.json"
    midi_path = midi_output_dir / "bass_ai_completion.mid"

    model_name = _resolve_openai_model(params.model)

    try:
        pattern_pack_result = build_pattern_pack(project_dir=project_dir, layer=params.layer)
    except PatternPackBuildError as exc:
        report = _error_report(
            project_dir=project_dir,
            layer=params.layer,
            model_name=model_name,
            pattern_pack_path=pattern_pack_path,
            ai_request_pack_path=ai_request_pack_path,
            prompt_path=prompt_path,
            ai_json_path=ai_json_path,
            midi_path=midi_path,
            dry_run=params.dry_run,
            message=str(exc),
            api_key_source="env",
            base_note_source=None,
        )
        _write_report(report_path, report)
        raise AIPatternCompletionError(str(exc)) from exc

    pattern_pack_path.write_text(
        json.dumps(pattern_pack_result.pattern_pack, indent=2) + "\n",
        encoding="utf-8",
    )

    ai_request_pack = build_ai_request_pack(pattern_pack_result.pattern_pack)
    ai_request_pack_path.write_text(
        json.dumps(ai_request_pack, indent=2) + "\n",
        encoding="utf-8",
    )

    system_prompt, user_prompt, combined_prompt = build_ai_completion_prompts(
        ai_request_pack=ai_request_pack,
        max_completion_notes=params.max_completion_notes,
    )
    prompt_path.write_text(combined_prompt + "\n", encoding="utf-8")

    full_pattern_pack_size_bytes = pattern_pack_path.stat().st_size
    ai_request_pack_size_bytes = ai_request_pack_path.stat().st_size
    ai_prompt_size_bytes = len(combined_prompt.encode("utf-8"))

    warnings = list(pattern_pack_result.warnings)
    if len(user_prompt) > _MAX_AI_PROMPT_CHARS or len(combined_prompt) > _MAX_AI_PROMPT_CHARS:
        message = (
            "AI request pack is too large for model context. "
            f"Compact pack size: {ai_request_pack_size_bytes} bytes."
        )
        report = _error_report(
            project_dir=project_dir,
            layer=params.layer,
            model_name=model_name,
            pattern_pack_path=pattern_pack_path,
            ai_request_pack_path=ai_request_pack_path,
            prompt_path=prompt_path,
            ai_json_path=ai_json_path,
            midi_path=midi_path,
            dry_run=params.dry_run,
            message=message,
            warnings=warnings,
            api_key_source="env",
            base_note_source=pattern_pack_result.base_note_source,
            full_pattern_pack_size_bytes=full_pattern_pack_size_bytes,
            ai_request_pack_size_bytes=ai_request_pack_size_bytes,
            ai_prompt_size_bytes=ai_prompt_size_bytes,
        )
        _write_report(report_path, report)
        raise AIPatternCompletionError(message)

    if params.dry_run:
        report = AIPatternCompletionReport(
            status="ok",
            project_dir=str(project_dir),
            layer=params.layer,
            model=model_name,
            api_called=False,
            api_key_source="env",
            dry_run=True,
            pattern_pack_file=str(pattern_pack_path),
            full_pattern_pack_file=str(pattern_pack_path),
            ai_request_pack_file=str(ai_request_pack_path),
            ai_prompt_file=str(prompt_path),
            full_pattern_pack_size_bytes=full_pattern_pack_size_bytes,
            ai_request_pack_size_bytes=ai_request_pack_size_bytes,
            ai_prompt_size_bytes=ai_prompt_size_bytes,
            ai_json_file=None,
            output_midi_file=None,
            output_midi_path=None,
            base_note_source=pattern_pack_result.base_note_source,
            retry_count=0,
            retry_reason=None,
            first_pass_proposed_note_count=0,
            first_pass_rejected_reasons={},
            final_proposed_note_count=0,
            proposed_note_count=0,
            accepted_note_count=0,
            rejected_note_count=0,
            rejected_reasons={},
            pitch_range_used={"min": None, "max": None},
            warning_count=len(warnings),
            warnings=warnings,
            error=None,
            raw_response_text=None,
        )
        _write_report(report_path, report)
        return report

    api_key, api_key_source = _resolve_openai_api_key(project_dir)
    completion_client = ai_client or OpenAIPatternCompletionClient()

    raw_response_text: str | None = None
    parsed_payload: dict[str, object] | None = None
    first_pass_proposed_note_count = 0
    first_pass_rejected_reasons: dict[str, int] = {}
    retry_count = 0
    retry_reason: str | None = None

    try:
        first_raw_response_text, first_parsed_payload, first_ai_output = _request_ai_completion(
            completion_client=completion_client,
            api_key=api_key,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=params.temperature,
            max_completion_notes=params.max_completion_notes,
        )
    except OpenAIPatternCompletionClientError as exc:
        report = _error_report(
            project_dir=project_dir,
            layer=params.layer,
            model_name=model_name,
            pattern_pack_path=pattern_pack_path,
            ai_request_pack_path=ai_request_pack_path,
            prompt_path=prompt_path,
            ai_json_path=ai_json_path,
            midi_path=midi_path,
            dry_run=False,
            message=str(exc),
            raw_response_text=raw_response_text,
            warnings=warnings,
            api_key_source=api_key_source,
            base_note_source=pattern_pack_result.base_note_source,
            full_pattern_pack_size_bytes=full_pattern_pack_size_bytes,
            ai_request_pack_size_bytes=ai_request_pack_size_bytes,
            ai_prompt_size_bytes=ai_prompt_size_bytes,
        )
        _write_report(report_path, report)
        raise AIPatternCompletionError(str(exc)) from exc
    except ValidationError as exc:
        message = f"AI JSON schema validation failed: {exc.errors()[0]['msg']}"
        report = _error_report(
            project_dir=project_dir,
            layer=params.layer,
            model_name=model_name,
            pattern_pack_path=pattern_pack_path,
            ai_request_pack_path=ai_request_pack_path,
            prompt_path=prompt_path,
            ai_json_path=ai_json_path,
            midi_path=midi_path,
            dry_run=False,
            message=message,
            raw_response_text=raw_response_text,
            warnings=warnings,
            api_key_source=api_key_source,
            base_note_source=pattern_pack_result.base_note_source,
            full_pattern_pack_size_bytes=full_pattern_pack_size_bytes,
            ai_request_pack_size_bytes=ai_request_pack_size_bytes,
            ai_prompt_size_bytes=ai_prompt_size_bytes,
        )
        _write_report(report_path, report)
        raise AIPatternCompletionError(message) from exc

    first_pass_proposed_note_count = len(first_ai_output.notes)
    first_validation_result = validate_ai_completion_notes(
        ai_output=first_ai_output,
        base_notes=pattern_pack_result.base_notes,
        project_duration_sec=pattern_pack_result.duration_sec,
        max_completion_notes=params.max_completion_notes,
    )
    first_pass_rejected_reasons = dict(first_validation_result.rejected_reason_counts)

    final_ai_output = first_ai_output
    final_validation_result = first_validation_result
    raw_response_text = first_raw_response_text
    parsed_payload = first_parsed_payload

    if _should_retry_duplicate_feedback(
        validation_result=first_validation_result,
        proposed_note_count=first_pass_proposed_note_count,
    ):
        retry_count = 1
        retry_reason = (
            "First pass rejected notes were mostly duplicate_base_note_onset "
            "or duplicate_base_note_overlap."
        )

        feedback_context = _build_retry_feedback_context(
            ai_output=first_ai_output,
            validation_result=first_validation_result,
        )
        system_prompt, user_prompt, combined_prompt = build_ai_completion_prompts(
            ai_request_pack=ai_request_pack,
            max_completion_notes=params.max_completion_notes,
            feedback_context=feedback_context,
        )
        prompt_path.write_text(combined_prompt + "\n", encoding="utf-8")
        ai_prompt_size_bytes = len(combined_prompt.encode("utf-8"))

        if len(user_prompt) > _MAX_AI_PROMPT_CHARS or len(combined_prompt) > _MAX_AI_PROMPT_CHARS:
            message = (
                "AI request pack is too large for model context. "
                f"Compact pack size: {ai_request_pack_size_bytes} bytes."
            )
            report = _error_report(
                project_dir=project_dir,
                layer=params.layer,
                model_name=model_name,
                pattern_pack_path=pattern_pack_path,
                ai_request_pack_path=ai_request_pack_path,
                prompt_path=prompt_path,
                ai_json_path=ai_json_path,
                midi_path=midi_path,
                dry_run=False,
                message=message,
                raw_response_text=raw_response_text,
                warnings=warnings,
                api_key_source=api_key_source,
                base_note_source=pattern_pack_result.base_note_source,
                full_pattern_pack_size_bytes=full_pattern_pack_size_bytes,
                ai_request_pack_size_bytes=ai_request_pack_size_bytes,
                ai_prompt_size_bytes=ai_prompt_size_bytes,
                retry_count=retry_count,
                retry_reason=retry_reason,
                first_pass_proposed_note_count=first_pass_proposed_note_count,
                first_pass_rejected_reasons=first_pass_rejected_reasons,
                final_proposed_note_count=first_pass_proposed_note_count,
            )
            _write_report(report_path, report)
            raise AIPatternCompletionError(message)

        try:
            retry_raw_response_text, retry_parsed_payload, retry_ai_output = _request_ai_completion(
                completion_client=completion_client,
                api_key=api_key,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=params.temperature,
                max_completion_notes=params.max_completion_notes,
            )
        except OpenAIPatternCompletionClientError as exc:
            report = _error_report(
                project_dir=project_dir,
                layer=params.layer,
                model_name=model_name,
                pattern_pack_path=pattern_pack_path,
                ai_request_pack_path=ai_request_pack_path,
                prompt_path=prompt_path,
                ai_json_path=ai_json_path,
                midi_path=midi_path,
                dry_run=False,
                message=str(exc),
                raw_response_text=raw_response_text,
                warnings=warnings,
                api_key_source=api_key_source,
                base_note_source=pattern_pack_result.base_note_source,
                full_pattern_pack_size_bytes=full_pattern_pack_size_bytes,
                ai_request_pack_size_bytes=ai_request_pack_size_bytes,
                ai_prompt_size_bytes=ai_prompt_size_bytes,
                retry_count=retry_count,
                retry_reason=retry_reason,
                first_pass_proposed_note_count=first_pass_proposed_note_count,
                first_pass_rejected_reasons=first_pass_rejected_reasons,
                final_proposed_note_count=first_pass_proposed_note_count,
            )
            _write_report(report_path, report)
            raise AIPatternCompletionError(str(exc)) from exc
        except ValidationError as exc:
            message = f"AI JSON schema validation failed: {exc.errors()[0]['msg']}"
            report = _error_report(
                project_dir=project_dir,
                layer=params.layer,
                model_name=model_name,
                pattern_pack_path=pattern_pack_path,
                ai_request_pack_path=ai_request_pack_path,
                prompt_path=prompt_path,
                ai_json_path=ai_json_path,
                midi_path=midi_path,
                dry_run=False,
                message=message,
                raw_response_text=raw_response_text,
                warnings=warnings,
                api_key_source=api_key_source,
                base_note_source=pattern_pack_result.base_note_source,
                full_pattern_pack_size_bytes=full_pattern_pack_size_bytes,
                ai_request_pack_size_bytes=ai_request_pack_size_bytes,
                ai_prompt_size_bytes=ai_prompt_size_bytes,
                retry_count=retry_count,
                retry_reason=retry_reason,
                first_pass_proposed_note_count=first_pass_proposed_note_count,
                first_pass_rejected_reasons=first_pass_rejected_reasons,
                final_proposed_note_count=first_pass_proposed_note_count,
            )
            _write_report(report_path, report)
            raise AIPatternCompletionError(message) from exc

        retry_validation_result = validate_ai_completion_notes(
            ai_output=retry_ai_output,
            base_notes=pattern_pack_result.base_notes,
            project_duration_sec=pattern_pack_result.duration_sec,
            max_completion_notes=params.max_completion_notes,
        )

        final_ai_output = retry_ai_output
        final_validation_result = retry_validation_result
        raw_response_text = retry_raw_response_text
        parsed_payload = retry_parsed_payload

    if params.keep_ai_json and parsed_payload is not None:
        ai_json_path.write_text(json.dumps(parsed_payload, indent=2) + "\n", encoding="utf-8")

    if retry_count == 1:
        warnings.append(
            "Triggered duplicate-feedback retry because first pass mostly duplicated base MIDI notes."
        )
    warnings.extend(final_validation_result.warnings)
    if retry_count == 1 and not final_validation_result.accepted_notes:
        warnings.append("AI completion produced no accepted notes after duplicate-feedback retry.")

    export_ai_completion_midi(
        notes=final_validation_result.accepted_notes,
        output_midi_path=midi_path,
        ticks_per_beat=pattern_pack_result.ticks_per_beat,
        tempo_us_per_beat=pattern_pack_result.tempo_us_per_beat,
    )

    report = AIPatternCompletionReport(
        status="ok",
        project_dir=str(project_dir),
        layer=params.layer,
        model=model_name,
        api_called=True,
        api_key_source=api_key_source,
        dry_run=False,
        pattern_pack_file=str(pattern_pack_path),
        full_pattern_pack_file=str(pattern_pack_path),
        ai_request_pack_file=str(ai_request_pack_path),
        ai_prompt_file=str(prompt_path),
        full_pattern_pack_size_bytes=full_pattern_pack_size_bytes,
        ai_request_pack_size_bytes=ai_request_pack_size_bytes,
        ai_prompt_size_bytes=ai_prompt_size_bytes,
        ai_json_file=(str(ai_json_path) if params.keep_ai_json else None),
        output_midi_file=str(midi_path),
        output_midi_path=str(midi_path),
        base_note_source=pattern_pack_result.base_note_source,
        retry_count=retry_count,
        retry_reason=retry_reason,
        first_pass_proposed_note_count=first_pass_proposed_note_count,
        first_pass_rejected_reasons=first_pass_rejected_reasons,
        final_proposed_note_count=len(final_ai_output.notes),
        proposed_note_count=len(final_ai_output.notes),
        accepted_note_count=len(final_validation_result.accepted_notes),
        rejected_note_count=len(final_validation_result.rejected_notes),
        rejected_reasons=final_validation_result.rejected_reason_counts,
        pitch_range_used=final_validation_result.pitch_range_used,
        warning_count=len(warnings),
        warnings=warnings,
        error=None,
        raw_response_text=raw_response_text,
    )
    _write_report(report_path, report)
    return report


def _resolve_output_dir(project_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return project_dir / "midi" / "ai"
    if output_dir.is_absolute():
        return output_dir
    return project_dir / output_dir


def _resolve_openai_model(model_override: str | None) -> str:
    if model_override is not None and model_override.strip():
        return model_override.strip()
    value = os.environ.get("OPENAI_MODEL", "").strip()
    if value:
        return value
    return "gpt-4o-mini"


def _resolve_openai_api_key(project_dir: Path) -> tuple[str, str]:
    # Preserve existing process value for source detection, then let dotenv override stale env.
    before = os.environ.get("OPENAI_API_KEY", "").strip()
    loaded_from_dotenv = False
    loaded_from_dotenv = _load_dotenv_if_exists(Path.cwd() / ".env") or loaded_from_dotenv
    loaded_from_dotenv = _load_dotenv_if_exists(project_dir / ".env") or loaded_from_dotenv
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise AIPatternCompletionError(
            "OPENAI_API_KEY is missing. Create .env from .env.example or set the environment variable."
        )
    if loaded_from_dotenv and api_key != before:
        return api_key, "dotenv"
    if loaded_from_dotenv and not before:
        return api_key, "dotenv"
    return api_key, "env"


def _load_dotenv_if_exists(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False

    loaded = False

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if value.startswith('"') and value.endswith('"') and len(value) >= 2:
            value = value[1:-1]
        if value.startswith("'") and value.endswith("'") and len(value) >= 2:
            value = value[1:-1]

        os.environ[key] = value
        loaded = True

    return loaded


def _request_ai_completion(
    *,
    completion_client: OpenAIPatternCompletionClient,
    api_key: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_completion_notes: int,
) -> tuple[str, dict[str, object], AIPatternCompletionOutput]:
    raw_response_text, parsed_payload = completion_client.complete_pattern(
        api_key=api_key,
        model=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_completion_notes=max_completion_notes,
    )
    ai_output = AIPatternCompletionOutput.model_validate(parsed_payload)
    return raw_response_text, parsed_payload, ai_output


def _should_retry_duplicate_feedback(
    *,
    validation_result: AICompletionValidationResult,
    proposed_note_count: int,
) -> bool:
    if proposed_note_count <= 0:
        return False
    if len(validation_result.accepted_notes) > 0:
        return False

    rejected_total = sum(validation_result.rejected_reason_counts.values())
    if rejected_total <= 0:
        return False

    duplicate_count = (
        validation_result.rejected_reason_counts.get("duplicate_base_note_onset", 0)
        + validation_result.rejected_reason_counts.get("duplicate_base_note_overlap", 0)
    )
    if duplicate_count <= 0:
        return False

    return (float(duplicate_count) / float(rejected_total)) >= 0.6


def _build_retry_feedback_context(
    *,
    ai_output: AIPatternCompletionOutput,
    validation_result: AICompletionValidationResult,
) -> str:
    lines: list[str] = [
        "The previous output was rejected during validation.",
        "Rejected reasons summary: "
        + json.dumps(validation_result.rejected_reason_counts, separators=(",", ":")),
        (
            "Regenerate the JSON. Do not place notes on existing base note onsets. "
            "Do not copy referenced notes. Only add genuinely new missing/continuation notes "
            "between existing base notes or after existing note endings."
        ),
        "Rejected note examples:",
    ]

    notes_by_id: dict[str, AIPatternCompletionNote] = {}
    for note in ai_output.notes:
        notes_by_id[note.note_id] = note

    for rejected in validation_result.rejected_notes[:6]:
        note = notes_by_id.get(rejected.note_id)
        if note is None:
            lines.append(f"- note_id={rejected.note_id}, reason={rejected.reason}")
            continue
        lines.append(
            "- "
            f"note_id={note.note_id}, "
            f"start_sec={note.start_sec:.6f}, "
            f"end_sec={note.end_sec:.6f}, "
            f"pitch_midi={note.pitch_midi}, "
            f"reason={rejected.reason}"
        )

    return "\n".join(lines)


def _error_report(
    *,
    project_dir: Path,
    layer: str,
    model_name: str,
    pattern_pack_path: Path,
    ai_request_pack_path: Path,
    prompt_path: Path,
    ai_json_path: Path,
    midi_path: Path,
    dry_run: bool,
    message: str,
    raw_response_text: str | None = None,
    warnings: list[str] | None = None,
    api_key_source: str = "env",
    base_note_source: str | None = None,
    full_pattern_pack_size_bytes: int = 0,
    ai_request_pack_size_bytes: int = 0,
    ai_prompt_size_bytes: int = 0,
    retry_count: int = 0,
    retry_reason: str | None = None,
    first_pass_proposed_note_count: int = 0,
    first_pass_rejected_reasons: dict[str, int] | None = None,
    final_proposed_note_count: int = 0,
) -> AIPatternCompletionReport:
    warning_list = list(warnings or [])
    warning_list.append(message)
    first_pass_rejected = dict(first_pass_rejected_reasons or {})

    return AIPatternCompletionReport(
        status="error",
        project_dir=str(project_dir),
        layer=layer,
        model=model_name,
        api_called=not dry_run,
        api_key_source=("dotenv" if api_key_source == "dotenv" else "env"),
        dry_run=dry_run,
        pattern_pack_file=str(pattern_pack_path),
        full_pattern_pack_file=str(pattern_pack_path),
        ai_request_pack_file=str(ai_request_pack_path),
        ai_prompt_file=str(prompt_path),
        full_pattern_pack_size_bytes=int(full_pattern_pack_size_bytes),
        ai_request_pack_size_bytes=int(ai_request_pack_size_bytes),
        ai_prompt_size_bytes=int(ai_prompt_size_bytes),
        ai_json_file=(str(ai_json_path) if ai_json_path.exists() else None),
        output_midi_file=(str(midi_path) if midi_path.exists() else None),
        output_midi_path=(str(midi_path) if midi_path.exists() else None),
        base_note_source=base_note_source,
        retry_count=int(retry_count),
        retry_reason=retry_reason,
        first_pass_proposed_note_count=int(first_pass_proposed_note_count),
        first_pass_rejected_reasons=first_pass_rejected,
        final_proposed_note_count=int(final_proposed_note_count),
        proposed_note_count=int(final_proposed_note_count),
        accepted_note_count=0,
        rejected_note_count=0,
        rejected_reasons={},
        pitch_range_used={"min": None, "max": None},
        warning_count=len(warning_list),
        warnings=warning_list,
        error=message,
        raw_response_text=raw_response_text,
    )


def _write_report(report_path: Path, report: AIPatternCompletionReport) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
