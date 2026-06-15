from __future__ import annotations

from collections import defaultdict
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
    OpenAIPatternCompletionResult,
    calculate_max_output_tokens,
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


@dataclass(frozen=True)
class _AICompletionRequestResult:
    raw_response_text: str
    parsed_payload: dict[str, object]
    ai_output: AIPatternCompletionOutput
    response_debug: dict[str, object]
    max_output_tokens_used: int


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
    allowed_regions_path = analysis_output_dir / "allowed_completion_regions.json"
    midi_path = midi_output_dir / "bass_ai_completion.mid"
    raw_response_first_pass_path = analysis_output_dir / "openai_raw_response_first_pass.txt"
    raw_response_retry_path = analysis_output_dir / "openai_raw_response_retry.txt"
    openai_debug_path = analysis_output_dir / "openai_response_debug.json"

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
    allowed_regions_payload = pattern_pack_result.pattern_pack.get("allowed_completion_regions")
    if isinstance(allowed_regions_payload, list):
        allowed_regions_path.write_text(
            json.dumps(allowed_regions_payload, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        allowed_regions_path.write_text("[]\n", encoding="utf-8")

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
            json_retry_count=0,
            json_retry_reason=None,
            raw_response_file=str(raw_response_first_pass_path),
            retry_raw_response_file=None,
            openai_response_status=None,
            openai_finish_reason=None,
            max_output_tokens_used=calculate_max_output_tokens(params.max_completion_notes),
        )
        _write_report(report_path, report)
        raise AIPatternCompletionError(message)

    if params.dry_run:
        region_reports = _build_region_reports(
            allowed_completion_regions=pattern_pack_result.allowed_completion_regions,
            ai_output=None,
            validation_result=None,
        )
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
            allowed_completion_region_count=len(pattern_pack_result.allowed_completion_regions),
            allowed_completion_regions_file=str(allowed_regions_path),
            notes_by_region={},
            region_reports=region_reports,
            json_retry_count=0,
            json_retry_reason=None,
            retry_count=0,
            retry_reason=None,
            first_pass_proposed_note_count=0,
            first_pass_rejected_reasons={},
            final_proposed_note_count=0,
            raw_response_file=None,
            retry_raw_response_file=None,
            openai_response_status=None,
            openai_finish_reason=None,
            max_output_tokens_used=calculate_max_output_tokens(params.max_completion_notes),
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

    max_output_tokens_used = calculate_max_output_tokens(params.max_completion_notes)
    raw_response_text: str | None = None
    parsed_payload: dict[str, object] | None = None
    raw_response_file = str(raw_response_first_pass_path)
    retry_raw_response_file: str | None = None
    openai_response_status: str | None = None
    openai_finish_reason: str | None = None
    json_retry_count = 0
    json_retry_reason: str | None = None
    first_pass_proposed_note_count = 0
    first_pass_rejected_reasons: dict[str, int] = {}
    retry_count = 0
    retry_reason: str | None = None
    openai_response_debug: dict[str, object] = {
        "attempts": [],
        "max_completion_notes": int(params.max_completion_notes),
        "temperature": float(params.temperature),
    }

    first_ai_output: AIPatternCompletionOutput
    first_request: _AICompletionRequestResult
    try:
        first_request = _request_ai_completion(
            completion_client=completion_client,
            api_key=api_key,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=params.temperature,
            max_completion_notes=params.max_completion_notes,
        )
        max_output_tokens_used = first_request.max_output_tokens_used
        openai_response_status, openai_finish_reason = _extract_status_and_finish(
            first_request.response_debug
        )
        _append_openai_debug_attempt(
            openai_response_debug,
            attempt_name="first_pass",
            response_debug=first_request.response_debug,
            max_output_tokens_used=first_request.max_output_tokens_used,
            error=None,
        )
        _write_openai_response_debug(openai_debug_path, openai_response_debug)
        _write_raw_response(raw_response_first_pass_path, first_request.raw_response_text)
        raw_response_text = first_request.raw_response_text
        parsed_payload = first_request.parsed_payload
        first_ai_output = first_request.ai_output
    except OpenAIPatternCompletionClientError as exc:
        if exc.raw_response_text is not None:
            raw_response_text = exc.raw_response_text
            _write_raw_response(raw_response_first_pass_path, exc.raw_response_text)

        error_max_output_tokens = _extract_max_output_tokens(exc.response_debug)
        if error_max_output_tokens is not None:
            max_output_tokens_used = error_max_output_tokens

        error_status, error_finish = _extract_status_and_finish(exc.response_debug)
        if error_status:
            openai_response_status = error_status
        if error_finish:
            openai_finish_reason = error_finish

        _append_openai_debug_attempt(
            openai_response_debug,
            attempt_name="first_pass",
            response_debug=exc.response_debug,
            max_output_tokens_used=error_max_output_tokens,
            error=str(exc),
        )
        _write_openai_response_debug(openai_debug_path, openai_response_debug)

        if exc.code != "invalid_json":
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
                json_retry_count=json_retry_count,
                json_retry_reason=json_retry_reason,
                raw_response_file=raw_response_file,
                retry_raw_response_file=retry_raw_response_file,
                openai_response_status=openai_response_status,
                openai_finish_reason=openai_finish_reason,
                max_output_tokens_used=max_output_tokens_used,
            )
            _write_report(report_path, report)
            raise AIPatternCompletionError(str(exc)) from exc

        json_retry_count = 1
        json_retry_reason = "First pass response was not valid JSON."
        retry_raw_response_file = str(raw_response_retry_path)

        feedback_context = _build_json_retry_feedback_context(raw_response_text=exc.raw_response_text)
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
                json_retry_count=json_retry_count,
                json_retry_reason=json_retry_reason,
                raw_response_file=raw_response_file,
                retry_raw_response_file=retry_raw_response_file,
                openai_response_status=openai_response_status,
                openai_finish_reason=openai_finish_reason,
                max_output_tokens_used=max_output_tokens_used,
            )
            _write_report(report_path, report)
            raise AIPatternCompletionError(message)

        try:
            first_request = _request_ai_completion(
                completion_client=completion_client,
                api_key=api_key,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=params.temperature,
                max_completion_notes=params.max_completion_notes,
            )
        except OpenAIPatternCompletionClientError as retry_exc:
            if retry_exc.raw_response_text is not None:
                raw_response_text = retry_exc.raw_response_text
                _write_raw_response(raw_response_retry_path, retry_exc.raw_response_text)

            retry_error_max_output_tokens = _extract_max_output_tokens(retry_exc.response_debug)
            if retry_error_max_output_tokens is not None:
                max_output_tokens_used = retry_error_max_output_tokens

            retry_error_status, retry_error_finish = _extract_status_and_finish(retry_exc.response_debug)
            if retry_error_status:
                openai_response_status = retry_error_status
            if retry_error_finish:
                openai_finish_reason = retry_error_finish

            _append_openai_debug_attempt(
                openai_response_debug,
                attempt_name="json_retry",
                response_debug=retry_exc.response_debug,
                max_output_tokens_used=retry_error_max_output_tokens,
                error=str(retry_exc),
            )
            _write_openai_response_debug(openai_debug_path, openai_response_debug)

            message = (
                "OpenAI response was not valid JSON after one repair retry."
                if retry_exc.code == "invalid_json"
                else str(retry_exc)
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
                json_retry_count=json_retry_count,
                json_retry_reason=json_retry_reason,
                raw_response_file=raw_response_file,
                retry_raw_response_file=retry_raw_response_file,
                openai_response_status=openai_response_status,
                openai_finish_reason=openai_finish_reason,
                max_output_tokens_used=max_output_tokens_used,
            )
            _write_report(report_path, report)
            raise AIPatternCompletionError(message) from retry_exc
        except ValidationError as retry_exc:
            message = f"AI JSON schema validation failed: {retry_exc.errors()[0]['msg']}"
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
                json_retry_count=json_retry_count,
                json_retry_reason=json_retry_reason,
                raw_response_file=raw_response_file,
                retry_raw_response_file=retry_raw_response_file,
                openai_response_status=openai_response_status,
                openai_finish_reason=openai_finish_reason,
                max_output_tokens_used=max_output_tokens_used,
            )
            _write_report(report_path, report)
            raise AIPatternCompletionError(message) from retry_exc

        max_output_tokens_used = first_request.max_output_tokens_used
        openai_response_status, openai_finish_reason = _extract_status_and_finish(
            first_request.response_debug
        )
        _append_openai_debug_attempt(
            openai_response_debug,
            attempt_name="json_retry",
            response_debug=first_request.response_debug,
            max_output_tokens_used=first_request.max_output_tokens_used,
            error=None,
        )
        _write_openai_response_debug(openai_debug_path, openai_response_debug)
        _write_raw_response(raw_response_retry_path, first_request.raw_response_text)
        raw_response_text = first_request.raw_response_text
        parsed_payload = first_request.parsed_payload
        first_ai_output = first_request.ai_output
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
            json_retry_count=json_retry_count,
            json_retry_reason=json_retry_reason,
            raw_response_file=raw_response_file,
            retry_raw_response_file=retry_raw_response_file,
            openai_response_status=openai_response_status,
            openai_finish_reason=openai_finish_reason,
            max_output_tokens_used=max_output_tokens_used,
        )
        _write_report(report_path, report)
        raise AIPatternCompletionError(message) from exc

    first_pass_proposed_note_count = len(first_ai_output.notes)
    first_validation_result = validate_ai_completion_notes(
        ai_output=first_ai_output,
        base_notes=pattern_pack_result.base_notes,
        project_duration_sec=pattern_pack_result.duration_sec,
        max_completion_notes=params.max_completion_notes,
        allowed_completion_regions=pattern_pack_result.allowed_completion_regions,
    )
    first_pass_rejected_reasons = dict(first_validation_result.rejected_reason_counts)

    final_ai_output = first_ai_output
    final_validation_result = first_validation_result

    if json_retry_count == 0 and _should_retry_duplicate_feedback(
        validation_result=first_validation_result,
        proposed_note_count=first_pass_proposed_note_count,
    ):
        retry_count = 1
        retry_reason = (
            "First pass rejected notes were mostly duplicate_base_note_onset "
            "or duplicate_base_note_overlap."
        )
        retry_raw_response_file = str(raw_response_retry_path)

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
                json_retry_count=json_retry_count,
                json_retry_reason=json_retry_reason,
                retry_count=retry_count,
                retry_reason=retry_reason,
                first_pass_proposed_note_count=first_pass_proposed_note_count,
                first_pass_rejected_reasons=first_pass_rejected_reasons,
                final_proposed_note_count=first_pass_proposed_note_count,
                raw_response_file=raw_response_file,
                retry_raw_response_file=retry_raw_response_file,
                openai_response_status=openai_response_status,
                openai_finish_reason=openai_finish_reason,
                max_output_tokens_used=max_output_tokens_used,
            )
            _write_report(report_path, report)
            raise AIPatternCompletionError(message)

        try:
            retry_request = _request_ai_completion(
                completion_client=completion_client,
                api_key=api_key,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=params.temperature,
                max_completion_notes=params.max_completion_notes,
            )
        except OpenAIPatternCompletionClientError as exc:
            if exc.raw_response_text is not None:
                raw_response_text = exc.raw_response_text
                _write_raw_response(raw_response_retry_path, exc.raw_response_text)

            retry_error_max_output_tokens = _extract_max_output_tokens(exc.response_debug)
            if retry_error_max_output_tokens is not None:
                max_output_tokens_used = retry_error_max_output_tokens

            retry_error_status, retry_error_finish = _extract_status_and_finish(exc.response_debug)
            if retry_error_status:
                openai_response_status = retry_error_status
            if retry_error_finish:
                openai_finish_reason = retry_error_finish

            _append_openai_debug_attempt(
                openai_response_debug,
                attempt_name="duplicate_retry",
                response_debug=exc.response_debug,
                max_output_tokens_used=retry_error_max_output_tokens,
                error=str(exc),
            )
            _write_openai_response_debug(openai_debug_path, openai_response_debug)

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
                json_retry_count=json_retry_count,
                json_retry_reason=json_retry_reason,
                retry_count=retry_count,
                retry_reason=retry_reason,
                first_pass_proposed_note_count=first_pass_proposed_note_count,
                first_pass_rejected_reasons=first_pass_rejected_reasons,
                final_proposed_note_count=first_pass_proposed_note_count,
                raw_response_file=raw_response_file,
                retry_raw_response_file=retry_raw_response_file,
                openai_response_status=openai_response_status,
                openai_finish_reason=openai_finish_reason,
                max_output_tokens_used=max_output_tokens_used,
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
                json_retry_count=json_retry_count,
                json_retry_reason=json_retry_reason,
                retry_count=retry_count,
                retry_reason=retry_reason,
                first_pass_proposed_note_count=first_pass_proposed_note_count,
                first_pass_rejected_reasons=first_pass_rejected_reasons,
                final_proposed_note_count=first_pass_proposed_note_count,
                raw_response_file=raw_response_file,
                retry_raw_response_file=retry_raw_response_file,
                openai_response_status=openai_response_status,
                openai_finish_reason=openai_finish_reason,
                max_output_tokens_used=max_output_tokens_used,
            )
            _write_report(report_path, report)
            raise AIPatternCompletionError(message) from exc

        max_output_tokens_used = retry_request.max_output_tokens_used
        openai_response_status, openai_finish_reason = _extract_status_and_finish(
            retry_request.response_debug
        )
        _append_openai_debug_attempt(
            openai_response_debug,
            attempt_name="duplicate_retry",
            response_debug=retry_request.response_debug,
            max_output_tokens_used=retry_request.max_output_tokens_used,
            error=None,
        )
        _write_openai_response_debug(openai_debug_path, openai_response_debug)
        _write_raw_response(raw_response_retry_path, retry_request.raw_response_text)

        retry_validation_result = validate_ai_completion_notes(
            ai_output=retry_request.ai_output,
            base_notes=pattern_pack_result.base_notes,
            project_duration_sec=pattern_pack_result.duration_sec,
            max_completion_notes=params.max_completion_notes,
            allowed_completion_regions=pattern_pack_result.allowed_completion_regions,
        )

        final_ai_output = retry_request.ai_output
        final_validation_result = retry_validation_result
        raw_response_text = retry_request.raw_response_text
        parsed_payload = retry_request.parsed_payload

    if params.keep_ai_json and parsed_payload is not None:
        ai_json_path.write_text(json.dumps(parsed_payload, indent=2) + "\n", encoding="utf-8")

    if retry_count == 1:
        warnings.append(
            "Triggered duplicate-feedback retry because first pass mostly duplicated base MIDI notes."
        )
    region_reports = _build_region_reports(
        allowed_completion_regions=pattern_pack_result.allowed_completion_regions,
        ai_output=final_ai_output,
        validation_result=final_validation_result,
    )
    if len(pattern_pack_result.allowed_completion_regions) == 0:
        warnings.append("No allowed completion regions were detected.")
    if final_validation_result.rejected_reason_counts.get("outside_allowed_completion_region", 0) > 0:
        warnings.append("AI attempted to place notes outside allowed completion regions.")
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
        allowed_completion_region_count=len(pattern_pack_result.allowed_completion_regions),
        allowed_completion_regions_file=str(allowed_regions_path),
        notes_by_region=final_validation_result.accepted_note_count_by_region,
        region_reports=region_reports,
        json_retry_count=json_retry_count,
        json_retry_reason=json_retry_reason,
        retry_count=retry_count,
        retry_reason=retry_reason,
        first_pass_proposed_note_count=first_pass_proposed_note_count,
        first_pass_rejected_reasons=first_pass_rejected_reasons,
        final_proposed_note_count=len(final_ai_output.notes),
        raw_response_file=raw_response_file,
        retry_raw_response_file=retry_raw_response_file,
        openai_response_status=openai_response_status,
        openai_finish_reason=openai_finish_reason,
        max_output_tokens_used=max_output_tokens_used,
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
) -> _AICompletionRequestResult:
    completion_result = completion_client.complete_pattern(
        api_key=api_key,
        model=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_completion_notes=max_completion_notes,
    )
    ai_output = AIPatternCompletionOutput.model_validate(completion_result.parsed_payload)
    return _AICompletionRequestResult(
        raw_response_text=completion_result.raw_response_text,
        parsed_payload=completion_result.parsed_payload,
        ai_output=ai_output,
        response_debug=completion_result.response_debug,
        max_output_tokens_used=completion_result.max_output_tokens_used,
    )


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


def _build_json_retry_feedback_context(*, raw_response_text: str | None) -> str:
    raw_snippet = (raw_response_text or "").strip()
    if len(raw_snippet) > 2000:
        raw_snippet = raw_snippet[:2000]

    lines = [
        "Your previous response was not valid JSON. Return JSON only matching the schema. No markdown, no comments, no prose.",
        "Invalid response excerpt (max 2000 chars):",
        raw_snippet,
    ]
    return "\n".join(lines)


def _write_raw_response(path: Path, raw_response_text: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((raw_response_text or "") + "\n", encoding="utf-8")


def _write_openai_response_debug(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _append_openai_debug_attempt(
    debug_payload: dict[str, object],
    *,
    attempt_name: str,
    response_debug: dict[str, object] | None,
    max_output_tokens_used: int | None,
    error: str | None,
) -> None:
    attempts_value = debug_payload.get("attempts")
    if not isinstance(attempts_value, list):
        attempts_value = []
        debug_payload["attempts"] = attempts_value

    attempt_payload: dict[str, object] = {
        "attempt": attempt_name,
        "max_output_tokens_used": int(max_output_tokens_used)
        if isinstance(max_output_tokens_used, int)
        else None,
        "response_debug": dict(response_debug or {}),
        "error": error,
    }
    attempts_value.append(attempt_payload)


def _extract_status_and_finish(response_debug: dict[str, object] | None) -> tuple[str | None, str | None]:
    payload = dict(response_debug or {})
    status = payload.get("status")
    finish_reason = payload.get("finish_reason")

    status_value = status if isinstance(status, str) and status.strip() else None
    finish_value = finish_reason if isinstance(finish_reason, str) and finish_reason.strip() else None
    return status_value, finish_value


def _extract_max_output_tokens(response_debug: dict[str, object] | None) -> int | None:
    payload = dict(response_debug or {})
    value = payload.get("max_output_tokens")
    if isinstance(value, int):
        return value
    return None


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
    allowed_completion_region_count: int = 0,
    allowed_completion_regions_file: str | None = None,
    notes_by_region: dict[str, int] | None = None,
    region_reports: list[dict[str, object]] | None = None,
    full_pattern_pack_size_bytes: int = 0,
    ai_request_pack_size_bytes: int = 0,
    ai_prompt_size_bytes: int = 0,
    json_retry_count: int = 0,
    json_retry_reason: str | None = None,
    retry_count: int = 0,
    retry_reason: str | None = None,
    first_pass_proposed_note_count: int = 0,
    first_pass_rejected_reasons: dict[str, int] | None = None,
    final_proposed_note_count: int = 0,
    raw_response_file: str | None = None,
    retry_raw_response_file: str | None = None,
    openai_response_status: str | None = None,
    openai_finish_reason: str | None = None,
    max_output_tokens_used: int = 0,
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
        allowed_completion_region_count=int(allowed_completion_region_count),
        allowed_completion_regions_file=allowed_completion_regions_file,
        notes_by_region=dict(notes_by_region or {}),
        region_reports=list(region_reports or []),
        json_retry_count=int(json_retry_count),
        json_retry_reason=json_retry_reason,
        retry_count=int(retry_count),
        retry_reason=retry_reason,
        first_pass_proposed_note_count=int(first_pass_proposed_note_count),
        first_pass_rejected_reasons=first_pass_rejected,
        final_proposed_note_count=int(final_proposed_note_count),
        raw_response_file=raw_response_file,
        retry_raw_response_file=retry_raw_response_file,
        openai_response_status=openai_response_status,
        openai_finish_reason=openai_finish_reason,
        max_output_tokens_used=int(max_output_tokens_used),
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


def _build_region_reports(
    *,
    allowed_completion_regions,
    ai_output: AIPatternCompletionOutput | None,
    validation_result: AICompletionValidationResult | None,
) -> list[dict[str, object]]:
    if not allowed_completion_regions:
        return []

    proposed_by_region: dict[str, int] = defaultdict(int)
    accepted_by_region: dict[str, int] = {}
    rejected_by_region: dict[str, int] = defaultdict(int)

    notes_by_id: dict[str, AIPatternCompletionNote] = {}
    if ai_output is not None:
        for note in ai_output.notes:
            notes_by_id[note.note_id] = note
            region_id = _find_region_id_for_timing(
                start_sec=float(note.start_sec),
                end_sec=float(note.end_sec),
                allowed_completion_regions=allowed_completion_regions,
            )
            if region_id is not None:
                proposed_by_region[region_id] += 1

    if validation_result is not None:
        accepted_by_region = dict(validation_result.accepted_note_count_by_region)
        for rejected in validation_result.rejected_notes:
            rejected_note = notes_by_id.get(rejected.note_id)
            if rejected_note is None:
                continue
            region_id = _find_region_id_for_timing(
                start_sec=float(rejected_note.start_sec),
                end_sec=float(rejected_note.end_sec),
                allowed_completion_regions=allowed_completion_regions,
            )
            if region_id is not None:
                rejected_by_region[region_id] += 1

    report_rows: list[dict[str, object]] = []
    for region in allowed_completion_regions:
        region_id = region.region_id
        proposed_count = int(proposed_by_region.get(region_id, 0))
        accepted_count = int(accepted_by_region.get(region_id, 0))
        rejected_count = int(rejected_by_region.get(region_id, 0))

        zero_reason: str | None = None
        if accepted_count == 0:
            if proposed_count == 0:
                if bool(region.optional_region):
                    zero_reason = "optional_or_low_confidence_region"
                elif int(region.expected_note_count_min) > 0:
                    zero_reason = "required_region_received_no_notes"
                else:
                    zero_reason = "no_notes_proposed"
            else:
                zero_reason = "all_region_notes_rejected"

        report_rows.append(
            {
                "region_id": region_id,
                "write_start_sec": round(float(region.write_start_sec), 6),
                "write_end_sec": round(float(region.write_end_sec), 6),
                "notes_before_count": len(region.notes_before),
                "notes_after_count": len(region.notes_after),
                "local_pitch_set": [int(value) for value in region.local_pitch_set],
                "detected_local_motif": dict(region.detected_local_motif),
                "motif_confidence": round(float(region.motif_confidence), 6),
                "optional_region": bool(region.optional_region),
                "ai_notes_proposed": proposed_count,
                "ai_notes_accepted": accepted_count,
                "ai_notes_rejected": rejected_count,
                "reason_if_zero_notes": zero_reason,
            }
        )

    return report_rows


def _find_region_id_for_timing(
    *,
    start_sec: float,
    end_sec: float,
    allowed_completion_regions,
) -> str | None:
    for region in allowed_completion_regions:
        write_start = float(getattr(region, "write_start_sec", region.start_sec))
        write_end = float(getattr(region, "write_end_sec", region.end_sec))
        if start_sec >= write_start and end_sec <= write_end:
            return region.region_id
    return None
