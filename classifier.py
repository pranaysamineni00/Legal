"""Legal clause classifier — Legal-BERT (CUAD) fine-tuned model with party-aware,
clause-aware LLM risk grading."""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── Clause metadata ──────────────────────────────────────────────────────────

CUAD_CLAUSES: list[str] = sorted([
    "Affiliate License-Licensee", "Affiliate License-Licensor",
    "Agreement Date", "Anti-Assignment", "Audit Rights",
    "Cap On Liability", "Change Of Control", "Competitive Restriction Exception",
    "Covenant Not To Sue", "Document Name", "Effective Date", "Exclusivity",
    "Expiration Date", "Governing Law",
    "Ip Ownership Assignment", "Insurance",
    "Irrevocable Or Perpetual License", "Joint Ip Ownership", "License Grant",
    "Liquidated Damages", "Minimum Commitment", "Most Favored Nation",
    "No-Solicit Of Customers", "No-Solicit Of Employees", "Non-Compete",
    "Non-Disparagement", "Non-Transferable License",
    "Notice Period To Terminate Renewal",
    "Parties", "Post-Termination Services", "Rofr/Rofo/Rofn", "Renewal Term",
    "Revenue/Profit Sharing", "Termination For Convenience",
    "Third Party Beneficiary", "Uncapped Liability", "Volume Restriction",
    "Warranty Duration",
])

# ── Party roles ──────────────────────────────────────────────────────────────
# Risk is graded for the named party only. `id` is what the API sends; `label`
# is what the UI shows.

PARTY_ROLES: list[dict[str, str]] = [
    {"id": "licensor",         "label": "Licensor (granting IP / software rights)"},
    {"id": "licensee",         "label": "Licensee (receiving IP / software rights)"},
    {"id": "service_provider", "label": "Service provider / vendor"},
    {"id": "customer",         "label": "Customer / client"},
    {"id": "buyer",            "label": "Buyer / purchaser"},
    {"id": "seller",           "label": "Seller / supplier"},
    {"id": "employer",         "label": "Employer / company"},
    {"id": "employee",         "label": "Employee / contractor"},
    {"id": "disclosing_party", "label": "Disclosing party (sharing confidential info)"},
    {"id": "receiving_party",  "label": "Receiving party (receiving confidential info)"},
    {"id": "generic",          "label": "Generic / unspecified party"},
]

PARTY_ROLE_IDS: set[str] = {r["id"] for r in PARTY_ROLES}


_PARTY_LABEL_BY_ID: dict[str, str] = {r["id"]: r["label"] for r in PARTY_ROLES}


def _party_label(role_id: str) -> str:
    return _PARTY_LABEL_BY_ID[role_id]


# ── Per-clause risk rubric ───────────────────────────────────────────────────
# Each entry tells the LLM grader: definition, factual features that move risk
# up/down, and how to map those features to HIGH/MEDIUM/LOW for the named party.
# The grader must apply this rubric verbatim — it cannot invent its own scale.

CLAUSE_RUBRICS: dict[str, dict[str, str]] = {
    "Affiliate License-Licensee": {
        "definition": "Extends the license to the licensee's affiliates (parent, subsidiaries, sister companies).",
        "factors": "Whether the named party is the licensor or licensee; breadth of affiliate definition; whether affiliates can sublicense further; revenue dilution if licensor.",
        "guidance": "If party is LICENSOR: HIGH when affiliate scope is open-ended (any entity under common control, future affiliates auto-included); MEDIUM when limited to existing majority-owned affiliates; LOW when narrow and revocable on divestiture. If party is LICENSEE: usually LOW (it is a benefit). For other parties: LOW unless they bear an indemnity for affiliate use.",
    },
    "Affiliate License-Licensor": {
        "definition": "Lets the licensor grant the licensed rights through its own affiliates, or have affiliates fulfill obligations.",
        "factors": "Which party the named party is; whether affiliates are jointly liable; assignment of affiliate-created IP back to licensor.",
        "guidance": "If party is LICENSEE: HIGH when affiliates can perform without licensor remaining responsible (no privity, no recourse); MEDIUM when licensor is jointly liable; LOW when affiliates are bound by the same terms with licensor on the hook. If party is LICENSOR: usually LOW. Generic: MEDIUM.",
    },
    "Agreement Date": {
        "definition": "The date the agreement is signed or dated.",
        "factors": "Pure metadata; never a substantive risk.",
        "guidance": "LOW for every party. Only flag MEDIUM if the date is materially inconsistent with the effective date in a way that creates retroactive obligations.",
    },
    "Anti-Assignment": {
        "definition": "Prohibits transferring the agreement (or rights/obligations under it) without consent.",
        "factors": "Which party is restricted; whether change-of-control is treated as assignment; whether consent must be reasonable; carve-outs for affiliate transfers.",
        "guidance": "HIGH for the named party when the restriction is one-sided against them, change-of-control triggers it, and consent may be withheld for any reason — this can block M&A. MEDIUM when mutual or consent must not be unreasonably withheld. LOW when the named party is the one whose consent is required, or carve-outs cover ordinary corporate restructuring.",
    },
    "Audit Rights": {
        "definition": "Right to inspect the counterparty's books and records.",
        "factors": "Which party holds the right vs. is subject to audit; frequency limits; cost-shifting on findings; scope (financial only vs. operational/security).",
        "guidance": "If party HOLDS the audit right: LOW. If party is SUBJECT to audit: HIGH when frequency is unlimited, scope is broad, and party pays regardless of findings; MEDIUM when capped to ~1x/year with reasonable notice; LOW when narrowly scoped and capped.",
    },
    "Cap On Liability": {
        "definition": "Caps each party's monetary liability under the contract.",
        "factors": "Cap amount relative to fees paid (multiple of annual fees, fixed dollar, etc.); mutuality; which damages are capped; carve-outs that escape the cap (IP, indemnity, confidentiality, gross negligence).",
        "guidance": "From the named party's perspective as a *plaintiff seeking recovery*: HIGH when cap is below 1x annual fees or one-sided against the named party (counterparty's exposure is limited but party's is not). MEDIUM when mutual at ~1x annual fees with standard carve-outs. LOW when generous (>2x) and mutual, or when the named party is the one the cap protects.",
    },
    "Change Of Control": {
        "definition": "Defines what happens on M&A or transfer of controlling interest (often termination, consent, or accelerated obligations).",
        "factors": "Which party's CoC is the trigger; consequences (termination right, payment acceleration, license revocation); whether competitor-acquirer carve-outs exist.",
        "guidance": "HIGH for the named party when their own CoC gives the counterparty an unconditional termination/revocation right with no transition period — this depresses M&A value. MEDIUM when CoC requires consent that cannot be unreasonably withheld. LOW when CoC has no automatic consequence or affects the counterparty only.",
    },
    "Competitive Restriction Exception": {
        "definition": "A carve-out from a non-compete or exclusivity obligation, permitting otherwise-restricted activity.",
        "factors": "Whether the carve-out benefits the named party or the counterparty; specificity of the carve-out.",
        "guidance": "Usually LOW or beneficial — it is a relief valve from a restriction. Bump to MEDIUM only if the carve-out is so broad that it neutralizes the named party's exclusivity (i.e., the named party paid for exclusivity and the exception destroys it).",
    },
    "Covenant Not To Sue": {
        "definition": "Promise by one party not to bring suit against another for specified claims.",
        "factors": "Which party is waiving the right to sue; scope of claims waived; whether the waiver covers future as well as past conduct; whether it survives termination.",
        "guidance": "HIGH for the party waiving (they are giving up legal recourse). LOW for the party benefiting. If mutual: MEDIUM. If the waiver covers gross negligence, willful misconduct, or IP infringement: HIGH regardless of mutuality.",
    },
    "Document Name": {
        "definition": "The title or naming of the agreement.",
        "factors": "Pure metadata.",
        "guidance": "LOW for every party.",
    },
    "Effective Date": {
        "definition": "The date the agreement's obligations begin.",
        "factors": "Whether the effective date is retroactive (creating obligations before signing); gap between signing and effectiveness.",
        "guidance": "LOW unless the effective date is materially retroactive and imposes obligations on the named party for periods predating signature — then MEDIUM.",
    },
    "Exclusivity": {
        "definition": "Restricts a party from engaging with competitors or alternative providers.",
        "factors": "Which party is restricted; geographic, product, and field-of-use scope; duration; carve-outs.",
        "guidance": "HIGH for the named party when they are the *restricted* party with broad scope (worldwide, all products, full term, no minimums in exchange). MEDIUM when scope is narrow (specific product line or region) or accompanied by minimum-purchase commitments from the counterparty. LOW when the named party is the *beneficiary* of the exclusivity.",
    },
    "Expiration Date": {
        "definition": "The date the agreement's term ends absent renewal.",
        "factors": "Term length; whether the date is fixed or formula-based.",
        "guidance": "LOW for every party. Only relevant in conjunction with renewal/termination provisions.",
    },
    "Governing Law": {
        "definition": "The jurisdiction whose law governs the contract.",
        "factors": "Whether the chosen jurisdiction is the named party's home jurisdiction or the counterparty's; whether forum selection is exclusive; arbitration vs. court.",
        "guidance": "LOW when the named party's home jurisdiction governs. MEDIUM when a neutral or industry-standard jurisdiction (e.g., New York, Delaware, English law) governs and the party has no home advantage. HIGH only when the chosen jurisdiction is materially unfavorable (counterparty's home, hostile to the party's industry, or imposes unusual procedural burdens).",
    },
    "Ip Ownership Assignment": {
        "definition": "Assigns ownership of intellectual property created under the contract.",
        "factors": "Who is the assignor (giving up IP) vs. assignee (receiving IP); what IP is assigned (work product only, or broader pre-existing IP and improvements); whether a license-back is granted to the assignor.",
        "guidance": "HIGH for the *assignor* — they are surrendering IP rights, often irrevocably. Especially HIGH if the scope sweeps in pre-existing IP, derivatives, or improvements without a license-back. LOW for the *assignee* (they receive the rights). MEDIUM if mutual cross-assignment or if a broad license-back leaves the assignor functionally able to continue using the IP.",
    },
    "Insurance": {
        "definition": "Required insurance coverage one party must maintain.",
        "factors": "Which party must carry insurance; coverage amounts; types (general liability, professional liability, cyber, workers' comp); whether the counterparty must be named as additional insured.",
        "guidance": "If the named party is REQUIRED to carry insurance: MEDIUM when amounts are within market norms for their industry; HIGH when amounts are atypically high or coverage types are difficult to obtain; LOW when modest. If the counterparty must carry: LOW for the named party.",
    },
    "Irrevocable Or Perpetual License": {
        "definition": "License that cannot be revoked (irrevocable) and/or runs forever (perpetual).",
        "factors": "Which party is licensor vs. licensee; scope of the perpetual/irrevocable grant; whether it survives termination for cause.",
        "guidance": "HIGH for the LICENSOR — they have permanently surrendered the right to claw back the license, even on the licensee's material breach. LOW for the LICENSEE (they have permanent rights). For non-IP parties: MEDIUM only if the license affects them.",
    },
    "Joint Ip Ownership": {
        "definition": "IP created jointly is co-owned by both parties.",
        "factors": "Whether each owner can exploit independently without accounting; whether either can license to third parties; jurisdictional default rules (US vs. EU treat joint ownership very differently).",
        "guidance": "MEDIUM for both parties by default — joint ownership creates ongoing coordination friction and can dilute commercial exclusivity. HIGH for the named party if the agreement is silent on independent exploitation rights and they operate in a jurisdiction where joint owners need consent to license (e.g., UK, Germany). LOW only when the agreement explicitly grants each owner full independent rights.",
    },
    "License Grant": {
        "definition": "The core grant of a license to use IP, software, or other rights.",
        "factors": "Scope (exclusive vs. non-exclusive); territory; field of use; permitted users; sublicense rights; revocability.",
        "guidance": "Risk depends on which side: for the LICENSEE, LOW when scope is broad and irrevocable, HIGH when narrowly defined with revocation triggers. For the LICENSOR, MEDIUM when granting exclusivity (even within a narrow field) since it forecloses other deals; LOW for non-exclusive grants. For non-IP parties: LOW unless they bear an indemnity.",
    },
    "Liquidated Damages": {
        "definition": "Pre-agreed monetary damages payable on specified breach.",
        "factors": "Which party owes the LD; amount relative to actual likely harm; whether it is the exclusive remedy or in addition to other remedies; whether it is structured as a penalty (often unenforceable).",
        "guidance": "HIGH for the named party if they are the one OWING liquidated damages and the amount is large relative to the contract value, or it is in addition to other remedies. MEDIUM when the LD is the exclusive remedy and amounts are proportionate to anticipated harm. LOW when the named party is the one ENTITLED to receive liquidated damages.",
    },
    "Minimum Commitment": {
        "definition": "Floor on purchase volume, spend, or activity that one party must meet.",
        "factors": "Which party owes the minimum; size relative to expected volumes; consequences of shortfall (true-up payment, termination right for counterparty, etc.); ramp / cure provisions.",
        "guidance": "HIGH for the OBLIGATED party when the floor is aggressive relative to forecast and shortfall triggers a true-up payment for the gap. MEDIUM when the floor is modest or the consequence is loss of exclusivity rather than out-of-pocket. LOW for the BENEFICIARY of the commitment.",
    },
    "Most Favored Nation": {
        "definition": "Promise that one party will not give other counterparties better terms.",
        "factors": "Which party owes the MFN; scope (all terms, just price, etc.); audit / enforcement mechanism; whether retroactive.",
        "guidance": "HIGH for the OBLIGATED party — MFNs constrain commercial flexibility, complicate every future deal, and are notoriously hard to manage. MEDIUM if narrowly scoped to price only and prospective. LOW for the BENEFICIARY.",
    },
    "No-Solicit Of Customers": {
        "definition": "Restriction on soliciting the counterparty's customers.",
        "factors": "Which party is restricted; duration; scope of customers covered (all vs. those introduced through the agreement); geographic scope.",
        "guidance": "HIGH for the RESTRICTED party when scope is broad, duration is long (>2 years post-termination), and 'customers' includes any party whose existence the restricted party learned of through the agreement. MEDIUM when limited to customers actually introduced through the deal and ≤1-year tail. LOW for the BENEFICIARY.",
    },
    "No-Solicit Of Employees": {
        "definition": "Restriction on hiring the counterparty's employees.",
        "factors": "Which party is restricted; duration; whether it covers solicitation only or any hiring (no-hire); carve-outs for general advertising or for employees who approach unsolicited.",
        "guidance": "HIGH for the RESTRICTED party when it is a no-hire (not just no-solicit), covers all employees not just those involved in the deal, and runs >2 years. MEDIUM for standard 1-year no-solicit with carve-outs. LOW for the BENEFICIARY.",
    },
    "Non-Compete": {
        "definition": "Restriction on engaging in competitive business or activity.",
        "factors": "Which party is restricted; activity scope; geographic scope; duration during and after the term; whether enforceable in the governing jurisdiction.",
        "guidance": "HIGH for the RESTRICTED party when scope covers their core business, is worldwide or industry-wide, and persists post-termination >1 year. MEDIUM when narrowly tailored (specific product line in specific region, ≤1 year tail) and supported by consideration. LOW for the BENEFICIARY.",
    },
    "Non-Disparagement": {
        "definition": "Restriction on making negative public statements about the counterparty.",
        "factors": "Which party is restricted; mutuality; carve-outs for truthful statements / regulatory disclosures / responses to subpoena; duration.",
        "guidance": "MEDIUM for the RESTRICTED party — these are common but constrain candor. HIGH if one-sided AND lacks carve-outs for truthful or legally compelled statements (a real free-speech / whistleblower risk). LOW for the BENEFICIARY or when fully mutual with standard carve-outs.",
    },
    "Non-Transferable License": {
        "definition": "License that cannot be transferred or sublicensed.",
        "factors": "Which party is the licensee; whether change-of-control is deemed a transfer; affiliate carve-outs.",
        "guidance": "HIGH for the LICENSEE when CoC is deemed a transfer (license dies on M&A). MEDIUM when CoC is not specifically addressed. LOW for the LICENSOR (it is a protective term for them) or when affiliate transfers are allowed.",
    },
    "Notice Period To Terminate Renewal": {
        "definition": "Required notice to opt out of automatic renewal.",
        "factors": "Length of notice required; whether failing to give notice locks in another full term; whether reminders are required.",
        "guidance": "MEDIUM for either party when notice is long (>90 days) and no reminder is required — easy to miss and lock into another term. LOW when ≤30 days and routine. HIGH only when missing notice locks in a multi-year term with no out.",
    },
    "Parties": {
        "definition": "Identification of the contracting parties.",
        "factors": "Pure metadata; risk only if the named legal entity is materially different from the operating entity (e.g., a thinly capitalized subsidiary).",
        "guidance": "LOW unless the counterparty is a shell or judgment-proof entity, in which case MEDIUM.",
    },
    "Post-Termination Services": {
        "definition": "Obligations to provide transition or wind-down services after termination.",
        "factors": "Which party owes the services; duration; payment terms during transition; data return / migration obligations.",
        "guidance": "HIGH for the OBLIGATED party when services must continue for >6 months post-termination at pre-termination pricing with no opt-out. MEDIUM when 90 days at uplift pricing is allowed. LOW for the BENEFICIARY (they receive a soft landing).",
    },
    "Rofr/Rofo/Rofn": {
        "definition": "Right of first refusal / first offer / first negotiation on future opportunities.",
        "factors": "Which party holds the right; scope of opportunities covered; duration; matching mechanics.",
        "guidance": "HIGH for the GRANTOR — these rights chill third-party deal-making for the encumbered asset. MEDIUM if narrowly scoped or short-duration. LOW for the HOLDER of the right.",
    },
    "Renewal Term": {
        "definition": "Defines whether and how the term renews.",
        "factors": "Auto-renew vs. opt-in; renewal length; price-escalation mechanics on renewal.",
        "guidance": "MEDIUM if auto-renew with no notice reminder and price-escalation cap is absent or weak — silent renewal is a classic trap. LOW for opt-in renewals or short auto-renew terms with capped escalation.",
    },
    "Revenue/Profit Sharing": {
        "definition": "One party's compensation tied to revenue, profit, or sales.",
        "factors": "Which party pays vs. receives; calculation base (gross/net); deductions allowed; audit rights; whether the rate steps down on milestones.",
        "guidance": "HIGH for the PAYER when the calculation base is broad (gross revenue with few deductions) and the rate is high. MEDIUM at typical industry rates on net revenue with standard deductions. LOW for the RECIPIENT.",
    },
    "Termination For Convenience": {
        "definition": "Right to terminate without cause on notice.",
        "factors": "Which party can terminate; notice period; whether unilateral or mutual; payment / wind-down obligations on termination.",
        "guidance": "HIGH for the named party when ONLY the counterparty has the right (asymmetric) and notice is short — investment in the relationship is at risk. MEDIUM when mutual with reasonable notice (≥60 days). LOW when only the named party holds the right.",
    },
    "Third Party Beneficiary": {
        "definition": "Whether non-parties can enforce the contract.",
        "factors": "Which third parties are named as beneficiaries; what rights they have; whether the named party has consented.",
        "guidance": "MEDIUM by default when third parties have enforcement rights — it expands the named party's exposure beyond the counterparty. HIGH when the third party is a competitor or regulator with broad enforcement standing. LOW when the agreement explicitly disclaims third-party beneficiaries.",
    },
    "Uncapped Liability": {
        "definition": "Liability that has no monetary cap, or carve-outs from a cap that effectively make exposure unlimited.",
        "factors": "Which party bears the uncapped exposure; scope of carve-outs (IP indemnity, confidentiality, gross negligence, fraud); mutuality.",
        "guidance": "HIGH for the party bearing uncapped exposure on broad categories (IP indemnity, data breach, confidentiality). MEDIUM if uncapped exposure is mutual or limited to narrow categories the named party fully controls (e.g., own gross negligence and willful misconduct only). LOW if the named party is the BENEFICIARY of unlimited recourse against the counterparty.",
    },
    "Volume Restriction": {
        "definition": "Cap on volume / quantity / activity.",
        "factors": "Which party is restricted; the cap level relative to expected demand; consequences of exceeding (overage fees, breach, termination right).",
        "guidance": "HIGH for the RESTRICTED party when the cap is below realistic forecast and overages are penalized as breach. MEDIUM when cap is reasonable and overages trigger only additional fees. LOW for the BENEFICIARY.",
    },
    "Warranty Duration": {
        "definition": "How long warranties remain in effect.",
        "factors": "Which party gives the warranty; length; whether it survives termination or acceptance.",
        "guidance": "If named party is the WARRANTOR: HIGH when warranties run >18 months post-acceptance or have no temporal limit. MEDIUM at standard 12 months. LOW for short (90-day) limited warranties. If named party is the BENEFICIARY: inverse — long warranties are LOW, short ones MEDIUM.",
    },
}

VALID_RISK_TIERS: tuple[str, ...] = ("HIGH", "MEDIUM", "LOW")

CATEGORIES: dict[str, list[str]] = {
    "Contract Basics":       ["Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date"],
    "Term & Duration":       ["Renewal Term", "Notice Period To Terminate Renewal", "Termination For Convenience", "Post-Termination Services"],
    "Financial & Liability": ["Cap On Liability", "Uncapped Liability", "Liquidated Damages", "Revenue/Profit Sharing", "Insurance", "Minimum Commitment", "Volume Restriction"],
    "Risk & Indemnity":      ["Covenant Not To Sue"],
    "IP & Licensing":        ["License Grant", "Ip Ownership Assignment", "Joint Ip Ownership", "Affiliate License-Licensor", "Affiliate License-Licensee", "Irrevocable Or Perpetual License", "Non-Transferable License"],
    "Restrictive Covenants": ["Non-Compete", "Non-Disparagement", "No-Solicit Of Customers", "No-Solicit Of Employees", "Exclusivity", "Competitive Restriction Exception"],
    "Contract Rights":       ["Anti-Assignment", "Change Of Control", "Audit Rights", "Rofr/Rofo/Rofn", "Third Party Beneficiary", "Most Favored Nation"],
    "Governing Law":         ["Governing Law"],
    "Warranties":            ["Warranty Duration"],
}


def _get_category(clause: str) -> str:
    for cat, members in CATEGORIES.items():
        if clause in members:
            return cat
    return "General"



_LLM_MAX_CHARS = 80000  # ~20K tokens — safe margin under gpt-4o-mini's 128K context


def _chunk_paragraphs(full_text: str, max_chars: int = _LLM_MAX_CHARS) -> list[str]:
    """Split text into chunks ≤ max_chars on paragraph boundaries."""
    if len(full_text) <= max_chars:
        return [full_text]

    def _hard_split(s: str) -> list[str]:
        return [s[i:i + max_chars] for i in range(0, len(s), max_chars)]

    chunks: list[str] = []
    cur = ""
    for para in full_text.split('\n\n'):
        if len(para) > max_chars:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.extend(_hard_split(para))
            continue
        if cur and len(cur) + len(para) + 2 > max_chars:
            chunks.append(cur)
            cur = para
        else:
            cur = (cur + '\n\n' + para) if cur else para
    if cur:
        chunks.append(cur)
    return chunks


def _grade_clause_via_llm(
    full_text: str,
    clause_name: str,
    party_role: str,
    client,
) -> dict:
    """Party-aware excerpt extraction + risk grading in one LLM call.

    Returns {"excerpt", "risk", "factors", "rationale"} or {} on failure.
    Large contracts are chunked on paragraph boundaries; the first chunk that
    yields a valid graded result wins.
    """
    rubric = CLAUSE_RUBRICS.get(clause_name)
    if rubric is None:
        logger.error("No rubric defined for clause '%s'.", clause_name)
        return {}

    party_label = _party_label(party_role)
    schema_block = (
        '{\n'
        '  "excerpt": "<verbatim passage, 1–6 sentences, copied character-for-character from the contract>",\n'
        '  "risk":    "HIGH" | "MEDIUM" | "LOW",\n'
        '  "factors": ["<short factual finding from the excerpt that drove the risk tier>", "..."],\n'
        '  "rationale": "<2–3 sentences explaining why this tier was chosen for ' + party_label + '>"\n'
        '}'
    )

    system_msg = (
        "You are a legal-risk analyst grading clauses in a commercial contract. "
        "You grade strictly from the perspective of one named party — never as an "
        "abstract property of the clause. You apply the rubric you are given verbatim "
        "and return a single JSON object that exactly matches the requested schema. "
        "You do not refuse, do not include markdown fences, and do not add commentary "
        "outside the JSON."
    )

    def _call(chunk: str) -> dict:
        user_prompt = (
            f"PARTY YOU REPRESENT: {party_label}\n"
            f"  (Risk must be graded from this party's perspective only.)\n\n"
            f"CLAUSE TYPE: {clause_name}\n"
            f"DEFINITION: {rubric['definition']}\n\n"
            f"RISK-DRIVING FACTORS (look for these in the excerpt):\n"
            f"  {rubric['factors']}\n\n"
            f"TIER TRIGGERS (apply verbatim — do NOT invent your own scale):\n"
            f"  {rubric['guidance']}\n\n"
            f"INSTRUCTIONS:\n"
            f"  1. Locate the passage in the CONTRACT below that establishes the {clause_name} clause.\n"
            f"  2. Quote it verbatim, 1–6 sentences, character-for-character.\n"
            f"  3. Identify which factual features from the FACTORS list appear in the excerpt.\n"
            f"  4. Apply the TIER TRIGGERS above to choose HIGH, MEDIUM, or LOW for {party_label}.\n"
            f"  5. Write a 2–3 sentence rationale grounded in the factual features you identified.\n"
            f"  6. Return ONE JSON object matching this schema exactly — no markdown, no fences, no commentary:\n\n"
            f"{schema_block}\n\n"
            f"CONTRACT:\n\"\"\"\n{chunk}\n\"\"\""
        )
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=900,
                response_format={"type": "json_object"},
            )
            content = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("LLM grading failed for '%s': %s", clause_name, exc)
            return {}

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.warning("LLM returned non-JSON for '%s': %s", clause_name, exc)
            return {}

        excerpt = (parsed.get("excerpt") or "").strip()
        risk = (parsed.get("risk") or "").strip().upper()
        factors = parsed.get("factors") or []
        rationale = (parsed.get("rationale") or "").strip()

        if risk not in VALID_RISK_TIERS:
            logger.warning("LLM returned invalid risk tier '%s' for '%s'.", risk, clause_name)
            return {}
        if not isinstance(factors, list):
            factors = [str(factors)]
        factors = [str(f).strip() for f in factors if str(f).strip()]
        if not excerpt:
            return {}
        if excerpt.startswith('"') and excerpt.endswith('"') and len(excerpt) > 1:
            excerpt = excerpt[1:-1].strip()

        return {
            "excerpt":   excerpt,
            "risk":      risk,
            "factors":   factors,
            "rationale": rationale,
        }

    for chunk in _chunk_paragraphs(full_text):
        result = _call(chunk)
        if result:
            return result
    return {}


def _trim_snippet(snippet: str, max_chars: int = 800) -> str:
    """Trim a snippet to max_chars, preferring sentence boundaries over word boundaries."""
    if len(snippet) <= max_chars:
        return snippet
    trimmed = snippet[:max_chars]
    sentence_cut = max(
        trimmed.rfind('. '), trimmed.rfind('.\n'),
        trimmed.rfind('? '), trimmed.rfind('! '),
    )
    if sentence_cut > max_chars - 250:
        return trimmed[:sentence_cut + 1].rstrip() + ' …'
    word_cut = trimmed.rfind(' ')
    if word_cut > max_chars - 100:
        trimmed = trimmed[:word_cut]
    return trimmed.rstrip() + '…'


# ── Main classifier ──────────────────────────────────────────────────────────

class LegalClauseClassifier:
    """Legal-BERT (CUAD) detector + party-aware LLM risk grader.

    Detection: the fine-tuned Legal-BERT checkpoint over sliding windows.
    Grading: per-clause LLM call applying CLAUSE_RUBRICS for the named party_role.
    """

    def __init__(self, checkpoint_path: str | None = None) -> None:
        self.mode = "model"
        self.model = None
        self.tokenizer = None
        self.id_to_clause: dict[int, str] = {}
        self.thresholds: dict[str, float] = {}
        self._device = None

        if checkpoint_path is None:
            checkpoint_path = str(Path(__file__).parent / "models" / "Legal-BERT_(CUAD).pt")

        self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, path: str) -> None:
        try:
            import pickle
            import torch
            from training import ModelArtifacts, _TfIdfPipeline  # noqa: F401

            device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
            art = torch.load(path, map_location=device, weights_only=False)
            if hasattr(art.model, "to"):
                art.model = art.model.to(device)
            art.model.eval()

            self.model = art.model
            self.tokenizer = art.tokenizer
            self.id_to_clause = art.id_to_clause
            self._device = device

            # Load per-clause thresholds from s4_outputs.pkl (produced by the
            # notebook's Section 4). Without this file the deployed app cannot
            # reproduce the macro-F1 reported in the notebook, since per-clause
            # thresholds (e.g. Document Name 0.05, Cap On Liability 0.90) are
            # not encoded in the model checkpoint.
            s4_path = Path(__file__).parent / "s4_outputs.pkl"
            per_clause_thresholds: dict[str, float] = {}
            if s4_path.exists():
                try:
                    with s4_path.open("rb") as f:
                        s4 = pickle.load(f)
                    raw = s4.get("per_t_best") or {}
                    if isinstance(raw, dict):
                        per_clause_thresholds = {str(k): float(v) for k, v in raw.items()}
                    logger.info(
                        "Loaded %d per-clause thresholds from %s.",
                        len(per_clause_thresholds), s4_path,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to read %s: %s — falling back to t=0.5.",
                        s4_path, exc,
                    )

            if not per_clause_thresholds:
                logger.warning(
                    "%s missing or empty — using t=0.5 for every clause. "
                    "Predictions will not match the notebook's reported macro-F1 "
                    "until s4_outputs.pkl is supplied.",
                    s4_path,
                )

            self.thresholds = {
                name: per_clause_thresholds.get(name, 0.5)
                for name in art.id_to_clause.values()
            }

            logger.info("Loaded Legal-BERT checkpoint from %s.", path)

        except Exception as exc:
            raise RuntimeError(f"Checkpoint load failed: {exc}") from exc

    def _get_openai_client(self):
        """Return a cached OpenAI client, or None if OPENAI_API_KEY is unset."""
        if hasattr(self, "_openai_client"):
            return self._openai_client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set — risk grading will be skipped.")
            self._openai_client = None
            return None
        from openai import OpenAI
        self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

    def _detect_clauses(self, text: str) -> list[tuple[str, float]]:
        """Run Legal-BERT over sliding windows; return (clause_name, prob) for
        every clause whose max window probability exceeds its threshold."""
        import torch

        enc = self.tokenizer(
            text,
            max_length=512,
            stride=128,
            truncation=True,
            padding="max_length",
            return_overflowing_tokens=True,
        )

        n_windows = len(enc["input_ids"])
        n_labels = len(self.id_to_clause)
        window_probs = np.zeros((n_windows, n_labels), dtype=np.float32)

        is_longformer = type(self.model).__name__.startswith("Longformer")
        for i in range(n_windows):
            ids = torch.tensor(enc["input_ids"][i:i+1], dtype=torch.long).to(self._device)
            mask = torch.tensor(enc["attention_mask"][i:i+1], dtype=torch.long).to(self._device)
            inputs: dict = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in enc:
                inputs["token_type_ids"] = torch.tensor(
                    enc["token_type_ids"][i:i+1], dtype=torch.long
                ).to(self._device)
            if is_longformer:
                gam = torch.zeros_like(ids)
                gam[:, 0] = 1
                inputs["global_attention_mask"] = gam

            with torch.no_grad():
                logits = self.model(**inputs).logits.cpu().numpy()[0]

            window_probs[i] = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))

        max_probs = window_probs.max(axis=0) if n_windows else np.zeros(n_labels, dtype=np.float32)

        detected: list[tuple[str, float]] = []
        for label_id, clause_name in self.id_to_clause.items():
            t = self.thresholds.get(clause_name, 0.5)
            prob = float(max_probs[label_id])
            if prob >= t:
                detected.append((clause_name, prob))
        return detected

    def _classify_model(self, text: str, party_role: str) -> list[dict]:
        """Detect every CUAD clause in `text` and grade each one for `party_role`.

        Detection is the BERT model; grading is one LLM call per detected clause
        that returns excerpt + risk + factors + rationale per CLAUSE_RUBRICS.
        If the LLM is unavailable, risk is None for every clause."""
        detected = self._detect_clauses(text)
        client = self._get_openai_client()

        def _build(clause_name: str, prob: float) -> dict:
            base = {
                "clause":     clause_name,
                "confidence": round(prob, 3),
                "category":   _get_category(clause_name),
                "party_role": party_role,
                "detected":   True,
            }
            if client is None:
                base.update({
                    "risk":      None,
                    "snippet":   "",
                    "factors":   [],
                    "rationale": "Risk not graded — OPENAI_API_KEY is not configured. The clause was detected by the neural model, but party-aware risk grading requires the LLM grader.",
                })
                return base

            graded = _grade_clause_via_llm(text, clause_name, party_role, client)
            if not graded:
                base.update({
                    "risk":      None,
                    "snippet":   "",
                    "factors":   [],
                    "rationale": "Risk not graded — the LLM grader did not return a valid response for this clause.",
                })
                return base

            base.update({
                "risk":      graded["risk"],
                "snippet":   _trim_snippet(graded["excerpt"]),
                "factors":   graded["factors"],
                "rationale": graded["rationale"],
            })
            return base

        if client is not None and detected:
            with ThreadPoolExecutor(max_workers=min(8, len(detected))) as pool:
                results = list(pool.map(lambda args: _build(*args), detected))
        else:
            results = [_build(name, prob) for name, prob in detected]

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    def classify(self, text: str, party_role: str = "generic") -> dict:
        clauses = self._classify_model(text, party_role)
        high = medium = low = ungraded = 0
        for c in clauses:
            risk = c["risk"]
            if risk == "HIGH":
                high += 1
            elif risk == "MEDIUM":
                medium += 1
            elif risk == "LOW":
                low += 1
            else:
                ungraded += 1
        return {
            "mode":         self.mode,
            "party_role":   party_role,
            "party_label":  _party_label(party_role),
            "total":        len(clauses),
            "risk_summary": {"HIGH": high, "MEDIUM": medium, "LOW": low, "UNGRADED": ungraded},
            "clauses":      clauses,
        }
