import re

# Basic format check (example: enforce bullet points or structured output)
def enforce_format(response):
    if not response.strip().startswith("-"):
        return False, "Response must start with bullet points."
    return True, "Format OK"

# Check for speculative language
SPECULATIVE_PATTERNS = [
    r"might have",
    r"could be",
    r"possibly",
    r"likely has",
    r"it seems that"
]

def detect_speculation(response):
    for pattern in SPECULATIVE_PATTERNS:
        if re.search(pattern, response, re.IGNORECASE):
            return True, f"Speculative language detected: '{pattern}'"
    return False, "No speculation detected"

# Combined guardrail check
def apply_guardrails(response):
    format_ok, format_msg = enforce_format(response)
    speculative, spec_msg = detect_speculation(response)

    result = {
        "format_pass": format_ok,
        "format_msg": format_msg,
        "speculation_flag": speculative,
        "speculation_msg": spec_msg,
        "safe_to_use": format_ok and not speculative
    }
    return result

# Example usage
def test_guardrails():
    test_response = "This patient might have early-stage hypertension."
    result = apply_guardrails(test_response)
    print("\n[Guardrail Check Result]")
    for key, value in result.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    test_guardrails()
