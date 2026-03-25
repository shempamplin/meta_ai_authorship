#!/usr/bin/env python3
"""Generate a peer review of a paper using an Anthropic persona profile."""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime

# Ensure pdfplumber is available
try:
    import pdfplumber
except ImportError:
    print("pdfplumber not found. Installing via pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfplumber"])
    import pdfplumber

import anthropic


def parse_persona(text: str) -> dict:
    """Extract the four persona sections from the persona file text."""
    headings = [
        ("analytical_sequence", "ANALYTICAL SEQUENCE"),
        ("epistemic_standards", "EPISTEMIC STANDARDS"),
        ("modality_markers", "EPISTEMIC MODALITY MARKERS"),
        ("ideological_substrate", "IDEOLOGICAL SUBSTRATE ELEMENT"),
    ]

    # Build a regex that captures text between known headings (or to end of string)
    all_heading_names = [h[1] for h in headings]
    sections = {}
    for key, heading in headings:
        # Match the heading line, then capture everything until the next known
        # heading or end of string.
        others = [re.escape(h) for h in all_heading_names if h != heading]
        if others:
            pattern = re.escape(heading) + r"[^\S\n]*\n(.*?)(?=" + "|".join(others) + r"|\Z)"
        else:
            pattern = re.escape(heading) + r"[^\S\n]*\n(.*)"
        match = re.search(pattern, text, re.DOTALL)
        sections[key] = match.group(1).strip() if match else ""

    return sections


def extract_paper_text(path: str) -> str:
    """Read paper text from a PDF or txt file."""
    if path.lower().endswith(".pdf"):
        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
        return "\n".join(pages)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


def main():
    parser = argparse.ArgumentParser(description="Generate a peer review using an Anthropic persona.")
    parser.add_argument("--persona", required=True, help="Path to persona txt file")
    parser.add_argument("--paper", required=True, help="Path to target paper (PDF or txt)")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.persona):
        print(f"Error: persona file not found: {args.persona}")
        sys.exit(1)
    if not os.path.isfile(args.paper):
        print(f"Error: paper file not found: {args.paper}")
        sys.exit(1)

    # Read persona
    with open(args.persona, "r", encoding="utf-8") as f:
        persona_text = f.read()

    sections = parse_persona(persona_text)

    # Extract paper text
    try:
        paper_text = extract_paper_text(args.paper)
    except Exception as e:
        print(f"Error reading paper file: {e}")
        sys.exit(1)

    # Build system prompt
    system_prompt = (
        "You are a peer reviewer. Your reviewing style, standards, and analytical approach "
        "are defined by the following profile.\n\n"
        f"EPISTEMIC STANDARDS: {sections['epistemic_standards']}\n\n"
        f"MODALITY MARKERS - use these phrases in the appropriate contexts: {sections['modality_markers']}\n\n"
        "IDEOLOGICAL SUBSTRATE - for each concern you raise, apply the two-question test: "
        "(1) Is the warrant empirical - the paper's evidence is insufficient by field standards? "
        "(2) Is the warrant ideological - the concern reflects a philosophical commitment not argued for? "
        "Label every concern as either [EMPIRICAL CONCERN] or [DOGMATIC CONCERN - Type X: one line description].\n"
        f"{sections['ideological_substrate']}\n\n"
        f"ANALYTICAL SEQUENCE - follow these steps in order when reviewing: {sections['analytical_sequence']}"
    )

    user_message = (
        "Please review the following paper according to your analytical sequence. "
        "For every concern you raise label it clearly as EMPIRICAL or DOGMATIC as instructed.\n\n"
        f"PAPER TEXT: {paper_text}"
    )

    # Call Anthropic API
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        review_text = response.content[0].text
    except Exception as e:
        print(f"Error calling Anthropic API: {e}")
        sys.exit(1)

    # Print review
    print(review_text)

    # Save to outputs directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(script_dir, "..", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"review_{timestamp}.txt"
    output_path = os.path.join(outputs_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(review_text)

    print(f"\nReview saved to: {output_path}")


if __name__ == "__main__":
    main()
