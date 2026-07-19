"""String utilities: text checks, transforms, and simple ciphers."""
import re


def is_palindrome(text: str) -> bool:
    """Return True if text reads the same forwards and backwards (ignoring case/spaces)."""
    cleaned = re.sub(r"[^a-zA-Z0-9]", "", text).lower()
    return cleaned == cleaned[::-1]


def slugify(text: str) -> str:
    """Convert text into a url-friendly slug, e.g. 'Hello World!' -> 'hello-world'."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def word_frequency(text: str) -> dict[str, int]:
    """Count occurrences of each word in text (case-insensitive)."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return freq


def caesar_cipher(text: str, shift: int) -> str:
    """Shift each letter in text by `shift` positions (Caesar cipher). Negative shift decodes."""
    result = []
    for ch in text:
        if ch.isalpha():
            base = ord('A') if ch.isupper() else ord('a')
            result.append(chr((ord(ch) - base + shift) % 26 + base))
        else:
            result.append(ch)
    return "".join(result)


def reverse_words(text: str) -> str:
    """Reverse the order of words in a sentence."""
    return " ".join(text.split()[::-1])


def levenshtein_distance(s1: str, s2: str) -> int:
    """Minimum number of single-character edits (insertions, deletions, substitutions)
    needed to turn s1 into s2."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def is_anagram(s1: str, s2: str) -> bool:
    """Return True if s1 and s2 use exactly the same letters (ignoring case/spaces)."""
    clean1 = re.sub(r"[^a-z0-9]", "", s1.lower())
    clean2 = re.sub(r"[^a-z0-9]", "", s2.lower())
    return sorted(clean1) == sorted(clean2)


def to_snake_case(text: str) -> str:
    """Convert 'camelCase' or 'Title Case' text into 'snake_case'."""
    text = re.sub(r"(?<!^)(?=[A-Z])", "_", text)
    text = re.sub(r"[\s\-]+", "_", text)
    return text.lower().strip("_")


def to_camel_case(text: str) -> str:
    """Convert 'snake_case' or 'kebab-case' text into 'camelCase'."""
    parts = re.split(r"[_\-\s]+", text.strip())
    if not parts:
        return ""
    return parts[0].lower() + "".join(p.capitalize() for p in parts[1:])


def rot13(text: str) -> str:
    """Apply the ROT13 cipher (Caesar cipher with a fixed shift of 13)."""
    return caesar_cipher(text, 13)


def vigenere_cipher(text: str, key: str, decode: bool = False) -> str:
    """Encode or decode text using the Vigenere cipher with the given key."""
    result = []
    key = key.lower()
    key_index = 0
    for ch in text:
        if ch.isalpha():
            shift = ord(key[key_index % len(key)]) - ord('a')
            if decode:
                shift = -shift
            base = ord('A') if ch.isupper() else ord('a')
            result.append(chr((ord(ch) - base + shift) % 26 + base))
            key_index += 1
        else:
            result.append(ch)
    return "".join(result)