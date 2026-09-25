"""Read immutable artifacts produced before the Nimble rename."""
from pathlib import Path

MODEL_NAME = "Bespoke-Nimble-9B"


def model_key(value):
    """Normalize a saved provider identifier without changing response contents."""
    return "nimble" if value == "openjeff" else value


def response_path(directory):
    directory = Path(directory)
    current = directory / "nimble_responses.jsonl"
    legacy = directory / "openjeff_responses.jsonl"
    return legacy if not current.exists() and legacy.exists() else current


def evaluation_settings(saved):
    """Normalize only the renamed metadata; retain all dataset fingerprints."""
    result = dict(saved)
    if "openjeff_model" in result:
        if "nimble_model" in result:
            raise ValueError("Conflicting legacy and current model settings")
        result["nimble_model"] = result.pop("openjeff_model")
    if result.get("nimble_model") == "bespoke-openjeff-9b":
        result["nimble_model"] = MODEL_NAME
    return result


def published_model_class(module):
    """Accept the frozen class name in existing downloaded model packages."""
    current = getattr(module, "NimbleModel", None)
    if current is not None:
        return current
    legacy = getattr(module, "OpenJeffModel", None)
    if legacy is not None:
        return legacy
    raise ImportError("Model package exports no supported Nimble inference class")
