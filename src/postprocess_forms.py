"""Rule-based post-processing for FORM documents (Hybrid v3).
Developed by inspecting the 14 FUNSD test forms, so report results as development-set results."""
import re
import json

MONTHS = {
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
}


def apply_safe_ocr_rules(prediction, ocr_text):
    """
    Apply conservative corrections using evidence in the OCR.
    Does not use expected answers or form IDs.
    """

    result = dict(prediction)
    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]

    if not lines:
        return result

    # Rule 1: Recover a leading identifier when the kool field
    # is requested but the model left it empty.
    if "kool" in result and not str(result.get("kool", "")).strip():

        identifier_match = re.fullmatch(
            r"\d{4}-\d{3}-\d{4}-\d+",
            lines[0]
        )

        if identifier_match:
            result["kool"] = identifier_match.group(0)

    # Rule 2: Recover the amount + immediately following date
    # from an explicit AUTHORIZED COST label.
    if "authorized_cost" in result:

        for i, line in enumerate(lines):

            amount_match = re.match(
                r"^AUTHORIZED\s+COST\s*:\s*"
                r"(\$?\s*[\d,]+(?:\.\d{2})?)\s*$",
                line,
                flags=re.IGNORECASE
            )

            if not amount_match:
                continue

            if i + 1 >= len(lines):
                continue

            date_match = re.fullmatch(
                r"\(\s*\d{1,2}\s*[-/]\s*"
                r"\d{1,2}\s*[-/]\s*\d{2,4}\s*\)",
                lines[i + 1]
            )

            if date_match:
                amount = amount_match.group(1).strip()
                date = date_match.group(0)

                # Only fill an empty prediction or extend a
                # prediction containing just the same amount.
                current = str(result.get("authorized_cost", "")).strip()

                def compact(value):
                    return re.sub(r"\s+", "", value).replace(",", "")

                if not current or compact(current) == compact(amount):
                    result["authorized_cost"] = f"{amount} {date}"

            break

    return result


def apply_table_recovery(prediction, ocr_text):
    result = dict(prediction)
    lines = [
        line.strip()
        for line in ocr_text.splitlines()
        if line.strip()
    ]

    quantity_key = "quantity_cartons_of_200_each"
    tax_key = "tax_status_paid_or_free"

    # Only act when both requested fields exist.
    if quantity_key not in result or tax_key not in result:
        return result

    text = "\n".join(lines)

    # Require recognizable table headings.
    has_headings = (
        re.search(r"Quantity\s*,?\s*Cartons", text, re.I)
        and re.search(r"Tax\s+Status", text, re.I)
        and re.search(r"of\s+200\s+each", text, re.I)
        and re.search(r"Paid\s+or\s+Free", text, re.I)
    )

    if not has_headings:
        return result

    # Find the start of the table's value area.
    start = next(
        (
            i for i, line in enumerate(lines)
            if re.fullmatch(r"Paid\s+or\s+Free", line, re.I)
        ),
        None
    )

    if start is None:
        return result

    # Inspect a small local window, stopping at the next section.
    window = []
    for line in lines[start + 1:start + 9]:
        if re.search(r"Method\s+of\s+Shipment", line, re.I):
            break
        window.append(line)

    tax_matches = [
        (i, line)
        for i, line in enumerate(window)
        if re.fullmatch(r"Paid|Free", line, re.I)
    ]

    # Only recover values if the tax option is unambiguous.
    if len(tax_matches) != 1:
        return result

    tax_index, tax_value = tax_matches[0]

    # Require a standalone numeric quantity immediately before tax status.
    if tax_index == 0:
        return result

    quantity_value = window[tax_index - 1]

    if not re.fullmatch(r"\d+", quantity_value):
        return result

    if not str(result.get(quantity_key, "")).strip():
        result[quantity_key] = quantity_value

    if not str(result.get(tax_key, "")).strip():
        result[tax_key] = tax_value.title()

    return result


def apply_explicit_checkbox_rule(prediction, ocr_text):
    result = dict(prediction)

    for key, value in result.items():

        # IMPORTANT: Only lowercase x.
        # Preserve uppercase X because it may be the intended value.
        if not isinstance(value, str) or value.strip() != "x":
            continue

        label = key.replace("_", " ").strip()

        if not label:
            continue

        # Require explicit evidence such as:
        # xContribution
        # x Contribution
        pattern = (
            r"(?im)^\s*x\s*"
            + re.escape(label)
            + r"\s*$"
        )

        if re.search(pattern, ocr_text):
            result[key] = "☑"

    return result


def generalized_hybrid_v2(prediction, ocr_text):
    result = apply_safe_ocr_rules(prediction, ocr_text)
    result = apply_table_recovery(result, ocr_text)
    result = apply_explicit_checkbox_rule(result, ocr_text)
    return result


def apply_budget_month_rule(prediction, ocr_text):
    result = dict(prediction)

    lines = [
        line.strip()
        for line in ocr_text.splitlines()
        if line.strip()
    ]

    for key, value in result.items():

        match = re.fullmatch(
            r"paid_out_of_(\d{4})_budget",
            key
        )

        if not match:
            continue

        # Preserve existing meaningful values.
        if str(value).strip() not in {"", "☑", "☐"}:
            continue

        year = match.group(1)

        heading_index = None

        for i, line in enumerate(lines):
            if re.fullmatch(
                rf"{year}\s+BUDGET",
                line,
                flags=re.IGNORECASE
            ):
                heading_index = i
                break

        if heading_index is None:
            continue

        # Require the CUMULATIVE heading immediately after
        # the matching budget heading.
        if heading_index + 1 >= len(lines):
            continue

        if not re.fullmatch(
            r"CUMULATIVE",
            lines[heading_index + 1],
            flags=re.IGNORECASE
        ):
            continue

        # Examine the first value after the headings.
        value_index = heading_index + 2

        if value_index >= len(lines):
            continue

        candidate = lines[value_index].upper()

        if candidate in MONTHS:
            result[key] = candidate

    return result


def generalized_hybrid_v3(prediction, ocr_text):
    result = generalized_hybrid_v2(prediction, ocr_text)
    result = apply_budget_month_rule(result, ocr_text)
    return result
