# src/preprocess.py
import re
import unicodedata
import json

def clean_text(text):
    """Unicode-normalize (Bengali-friendly) and collapse spaces. Accepts non-str input."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_texts(texts):
    """Apply clean_text to list of texts."""
    return [clean_text(t) for t in texts]

def _format_test_list(test_list):
    """Turn the test_list (list or string) into a readable block for the model."""
    if not test_list:
        return ""
    if isinstance(test_list, str):
        try:
            parsed = json.loads(test_list)
        except Exception:
            parsed = [test_list]
    else:
        parsed = test_list
    # Each test case on its own line (model sees examples / asserts)
    return "\n".join([str(x) for x in parsed])

def build_prompt(instruction, test_list=None):
    """
    Build a clear Bangla -> Python prompt that tells the model what's expected.
    The format is intentionally explicit so code LMs learn to respond with the function definition only.

    Example output:
    "# নির্দেশ: <bangla instruction>\n# টেস্টকেস:\n<tests>\n# কোড:\n"
    """
    instr = clean_text(instruction)
    tests = _format_test_list(test_list)
    prompt = []
    prompt.append("# নির্দেশনা:")  # "instruction" in Bangla
    prompt.append(instr)
    if tests:
        prompt.append("\n# টেস্টকেস (উদাহরণ/asserts):")
        prompt.append(tests)
    # Clear "output" marker — model should start code after this line
    prompt.append("\n# কোড (শুধু ফাংশন সংজ্ঞা লিখুন):\n")
    return "\n".join(prompt)
