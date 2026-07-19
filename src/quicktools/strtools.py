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

def longest_common_subsequence(s1: str, s2: str) -> str:
    """The longest sequence of characters that appears in both strings, in order (not necessarily contiguous)."""
    m, n = len(s1), len(s2)
    dp = [[""] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + s1[i - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], key=len)
    return dp[m][n]


def run_length_encode(text: str) -> str:
    """Compress text using run-length encoding, e.g. 'aaabbc' -> '3a2b1c'."""
    if not text:
        return ""
    result = []
    count = 1
    prev = text[0]
    for ch in text[1:]:
        if ch == prev:
            count += 1
        else:
            result.append(f"{count}{prev}")
            prev = ch
            count = 1
    result.append(f"{count}{prev}")
    return "".join(result)


def run_length_decode(encoded: str) -> str:
    """Decompress a run-length encoded string produced by run_length_encode()."""
    result = []
    count_str = ""
    for ch in encoded:
        if ch.isdigit():
            count_str += ch
        else:
            result.append(ch * int(count_str))
            count_str = ""
    return "".join(result)


def jaccard_similarity(s1: str, s2: str) -> float:
    """Word-level similarity between two texts: size of the word intersection over the union (0 to 1)."""
    words1 = set(re.findall(r"[a-zA-Z']+", s1.lower()))
    words2 = set(re.findall(r"[a-zA-Z']+", s2.lower()))
    if not words1 and not words2:
        return 1.0
    return len(words1 & words2) / len(words1 | words2)


def word_count(text: str) -> int:
    """Count the number of words in text."""
    return len(re.findall(r"[a-zA-Z']+", text))


def char_frequency(text: str) -> dict[str, int]:
    """Count occurrences of each character in text (case-sensitive, includes all characters)."""
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    return freq


def most_common_word(text: str) -> str:
    """Return the most frequently occurring word in text (case-insensitive)."""
    freq = word_frequency(text)
    if not freq:
        raise ValueError("Text contains no words")
    return max(freq, key=freq.get)


def title_case(text: str) -> str:
    """Capitalize the first letter of every word in text."""
    return " ".join(w.capitalize() for w in text.split())


def remove_vowels(text: str) -> str:
    """Remove all vowels (a, e, i, o, u) from text, case-insensitive."""
    return re.sub(r"[aeiouAEIOU]", "", text)


def count_vowels_consonants(text: str) -> tuple[int, int]:
    """Count the number of vowels and consonants in text (letters only)."""
    vowels = sum(1 for ch in text.lower() if ch in "aeiou")
    consonants = sum(1 for ch in text.lower() if ch.isalpha() and ch not in "aeiou")
    return vowels, consonants


def hash_text(text: str, algorithm: str = "sha256") -> str:
    """Return the hex digest of text using the given hash algorithm (e.g. 'md5', 'sha1', 'sha256')."""
    import hashlib
    h = hashlib.new(algorithm)
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def is_pangram(text: str) -> bool:
    """Return True if text contains every letter of the alphabet at least once."""
    letters = set(ch for ch in text.lower() if ch.isalpha())
    return len(letters) == 26


def wrap_text(text: str, width: int = 70) -> str:
    """Wrap text so no line exceeds `width` characters."""
    import textwrap
    return textwrap.fill(text, width=width)


def longest_word(text: str) -> str:
    """Return the longest word in text (first one found in case of a tie)."""
    words = re.findall(r"[a-zA-Z']+", text)
    if not words:
        raise ValueError("Text contains no words")
    return max(words, key=len)


def base64_encode(text: str) -> str:
    """Encode text into base64."""
    import base64
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def base64_decode(encoded: str) -> str:
    """Decode a base64-encoded string back into text."""
    import base64
    return base64.b64decode(encoded.encode("ascii")).decode("utf-8")