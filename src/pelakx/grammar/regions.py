"""Map an OCR engine's region guess onto an ISO-3166 alpha-2 country code.

``fast-plate-ocr`` predicts the issuing country of a plate alongside the text
(``region='Germany'``). That is a *prior*, not an answer: PelakX still requires
the string to satisfy that country's grammar before believing it. Used this
way it breaks ties in ``--country auto`` when two countries both parse the
same string — which happens constantly (``ABC1234`` is legal in a dozen
places).
"""

from __future__ import annotations

#: engine region label (lowercased) -> ISO-3166 alpha-2
REGION_TO_CODE: dict[str, str] = {
    "argentina": "AR",
    "brazil": "BR",
    "brasil": "BR",
    "chile": "CL",
    "colombia": "CO",
    "france": "FR",
    "germany": "DE",
    "deutschland": "DE",
    "india": "IN",
    "iran": "IR",
    "italy": "IT",
    "italia": "IT",
    "mexico": "MX",
    "netherlands": "NL",
    "nederland": "NL",
    "poland": "PL",
    "portugal": "PT",
    "romania": "RO",
    "russia": "RU",
    "spain": "ES",
    "espana": "ES",
    "turkey": "TR",
    "turkiye": "TR",
    "uae": "AE",
    "united arab emirates": "AE",
    "uk": "GB",
    "united kingdom": "GB",
    "great britain": "GB",
    "usa": "US",
    "united states": "US",
}


def region_to_code(region: str | None) -> str | None:
    """Best-effort ISO-2 code for an engine's region string."""
    if not region:
        return None
    key = region.strip().lower()
    if not key or key == "unknown":
        return None
    if len(key) == 2 and key.isalpha():
        return key.upper()
    return REGION_TO_CODE.get(key)
