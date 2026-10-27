#!/usr/bin/env python3
"""Build docs/wcag-demo/wcag_criteria.json: the 63 WCAG 2.2 criteria as vegaml
boolean questions, matching the criterion list LayaWCAG scores (its model/wcag_criteria.json).

Each entry: criterion, principle, level, description (LayaWCAG's), and `instructions` —
the sentence the vegaml reader reads with the pasted markup as state. The question is
boolean "violated?": a probability above the threshold in the demo means "violation found".
"""
import json
import pathlib

LAYA = pathlib.Path(__file__).resolve().parents[3] / "LayaWCAG" / "model" / "wcag_criteria.json"
OUT = pathlib.Path(__file__).parent / "wcag_criteria.json"

# Per-criterion intent, phrased for a markup-only reader: what a violation looks like
# in the HTML a user pastes. Kept short — the reader sees this as the question text.
INTENT = {
    "1.1.1": (
        "non-text content (img, icon, canvas, chart, emoji-as-symbol) has no alt text "
        "or equivalent text alternative"
    ),
    "1.2.1": "audio-only or video-only content has no equivalent alternative",
    "1.2.2": "prerecorded video has no captions",
    "1.2.3": "prerecorded audio has no audio description or media alternative",
    "1.2.4": "live video has no captions",
    "1.2.5": "prerecorded video has no audio description",
    "1.3.1": (
        "information and relationships (headings, tables, lists, form structure, labels) "
        "are not conveyed programmatically, only visually or positionally"
    ),
    "1.3.2": "the reading order in the markup does not match the meaningful sequence of the content",
    "1.3.3": "instructions rely only on sensory characteristics (shape, location, color) instead of text",
    "1.3.4": "content has a forced portrait or landscape orientation, or a fixed width",
    "1.3.5": (
        "input fields collect personal information but are not marked with autocomplete "
        "attributes so their purpose can be identified"
    ),
    "1.3.6": "the purpose of forms, form fields and controls cannot be programmatically identified",
    "1.4.1": "color is the only visual way to convey information or to operate something",
    "1.4.2": "audio autoplays for more than three seconds without a pause/stop/volume control",
    "1.4.3": (
        "text contrast is likely below 4.5:1 (foreground/background color values or a "
        "declared low-contrast theme are present)"
    ),
    "1.4.4": "the markup uses fixed units or rigid containers that would break when text is resized to 200%",
    "1.4.5": "an image of text is used to convey information instead of real text (not decorative)",
    "1.4.6": "body or heading text contrast is likely below 7:1 where an enhanced contrast level is required",
    "1.4.7": "background audio plays with speech content and is not stopped or substantially lower",
    "1.4.8": (
        "visual presentation locks in width, height, orientation or a required scrolling "
        "that breaks reflow"
    ),
    "1.4.9": "an image of text is used where even the exceptions do not apply",
    "1.4.10": "content in a fixed-width container would require horizontal scrolling at 320 CSS pixels",
    "1.4.11": "UI components and graphics need color to be understood but likely fall below 3:1 contrast",
    "1.4.12": "adjusting text spacing would cause loss of content or functionality",
    "1.4.13": (
        "information is conveyed only through hover content that disappears when the "
        "element loses hover"
    ),
    "2.1.1": (
        "functionality is operable with a pointer only (e.g. a custom drag interaction "
        "with no keyboard equivalent)"
    ),
    "2.1.2": (
        "a keyboard trap is present: focus can enter a component but cannot leave it "
        "with keyboard alone"
    ),
    "2.1.4": "a single character key shortcut (e.g. pressing 'f' to find) exists and cannot be turned off",
    "2.2.1": "time-limited content is presented without a way to extend, remove or adjust the time limit",
    "2.2.2": (
        "moving, blinking or auto-updating content lasts more than five seconds without "
        "pause, stop or hide controls"
    ),
    "2.3.1": "content flashes more than three times in one second or crosses the flash threshold",
    "2.4.1": (
        "there is no way to bypass repeated blocks: no landmark structure and no "
        "first-link-to-content pattern"
    ),
    "2.4.2": "the page has no title, or a title that is generic or missing",
    "2.4.3": "focusable elements receive focus in an order that does not preserve meaning and usability",
    "2.4.4": "links have text that is indistinguishable in context (e.g. every link says 'here' or 'click')",
    "2.4.5": (
        "there is only one way to find a page (no second navigation mechanism shown "
        "for multi-page content)"
    ),
    "2.4.6": "headings and labels do not describe the topic or purpose of their content",
    "2.4.7": "focus is not visible on any component, or the visible focus indicator is removed by styling",
    "2.4.11": (
        "focus is partially obscured by other content (a sticky footer/header or "
        "toolbar covers the focused element)"
    ),
    "2.4.12": "the entire focused component is obscured by non-essential content",
    "2.4.13": "the visible focus indicator is likely too thin or small to see (thin outlines, small padding)",
    "2.5.1": (
        "a multi-point or path-based gesture (pinch, rotate, two-finger swipe) has no "
        "single-pointer or keyboard alternative"
    ),
    "2.5.2": "a multi-point or path-based gesture cannot be cancelled and reversed",
    "2.5.3": "the visible label of an input or control is not part of its accessible name",
    "2.5.4": "an operation is triggered by device motion (shake, tilt) with no interface alternative",
    "2.5.5": "clickable target sizes are likely smaller than 44x44 CSS pixels",
    "2.5.6": (
        "a component requires a composite pointer+keyboard or multi-pointer+touch "
        "action with no single-mechanism alternative"
    ),
    "3.1.1": "the page has no lang attribute, or the lang does not match the content",
    "3.1.2": "text passages in a different language are not marked with their own lang attribute",
    "3.2.1": (
        "receiving focus automatically changes the context or adds new focusable "
        "content without warning"
    ),
    "3.2.2": (
        "changing a form control's value automatically submits the form or triggers a "
        "significant change without confirmation"
    ),
    "3.2.3": "the same user interface is not reused for the same function across these pages",
    "3.2.4": "components with the same function are not identified the same way across pages",
    "3.2.6": (
        "help (a help mechanism for this content) is not consistently available on "
        "request where the same task recurs across pages"
    ),
    "3.3.1": "an error is identified but not described in text (only color or icon)",
    "3.3.2": "form inputs have no visible label or instruction",
    "3.3.3": "an error is detected but no correction suggestion is offered",
    "3.3.4": "errors in important data are not reversible or no review/confirmation mechanism is present",
    "3.3.5": "contextual help for form errors is not available on request",
    "3.3.6": (
        "a significant legal/financial/medical entry cannot be reviewed, corrected or "
        "confirmed (no prevention)"
    ),
    "3.3.7": (
        "the same personal information is asked for again although it is already "
        "stored and can be auto-filled"
    ),
    "3.3.8": (
        "authentication depends only on a cognitive function (reading a math problem, "
        "remembering a sequence) without a cognitive alternative"
    ),
    "4.1.1": (
        "the markup does not parse: unclosed tags, duplicate IDs, overlapping ARIA "
        "roles, or ARIA used on non-focusable elements"
    ),
    "4.1.2": (
        "a control's name, role or value cannot be programmatically determined "
        "(custom widget without ARIA role/state)"
    ),
    "4.1.3": "status messages are not announced: they use a div without role=status or aria-live",
}


def main() -> None:
    laya = json.loads(LAYA.read_text())
    assert len(laya) == 63, len(laya)
    missing = [c["criterion"] for c in laya if c["criterion"] not in INTENT]
    assert not missing, missing
    out = []
    for c in laya:
        cid = c["criterion"]
        out.append({
            "criterion": cid,
            "principle": c["principle"],
            "level": c["level"],
            "description": c["description"],
            # The question the vegaml reader answers: p(true) = probability this markup
            # violates the criterion. Instructions carry the intent, so the zero-shot
            # engine has something discriminative to read.
            "instructions": (
                f"Does this HTML markup fail WCAG 2.2 criterion {cid} ({c['description'].lower()}), "
                f"i.e. it is violated because {INTENT[cid]}? Answer from the markup alone."
            ),
        })
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT} with {len(out)} criteria")


if __name__ == "__main__":
    main()
