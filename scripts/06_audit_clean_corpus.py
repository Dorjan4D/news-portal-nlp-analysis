#!/usr/bin/env python3
"""
Leakage / quality audit for the cleaned, near-deduplicated candidate corpus.

This is a PRE-SPLIT gate. Do not use the resulting classifier scores as final
research results. They are diagnostics for shortcuts/artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_validate
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


PORTALS = ["bbc", "guardian", "aljazeera"]
OUTLET_CHECKS = {
    "BBC": re.compile(r"\bBBC\b", re.IGNORECASE),
    "bbc.com/co.uk": re.compile(r"\b(?:www\.)?bbc\.(?:com|co\.uk)\b", re.IGNORECASE),
    "The Guardian": re.compile(r"\bThe Guardian\b", re.IGNORECASE),
    "theguardian.com": re.compile(r"\b(?:www\.)?theguardian\.com\b", re.IGNORECASE),
    "Al Jazeera": re.compile(r"\bAl\s*Jazeera\b", re.IGNORECASE),
    "aljazeera.com": re.compile(r"\b(?:www\.)?aljazeera\.com\b", re.IGNORECASE),
}
PROGRAM_CHECKS = {
    "Today programme": re.compile(r"\b(?:Radio\s+4(?:['’]s)?\s+)?Today\s+(?:programme|program)\b",re.I),
    "Newsnight": re.compile(r"\bNewsnight\b",re.I),
    "HARDtalk": re.compile(r"\bHARDtalk\b",re.I),
    "BBC Verify": re.compile(r"\bBBC\s+Verify\b",re.I),
    "BBC World Service": re.compile(r"\bBBC\s+World\s+Service\b",re.I),
    "BBC Panorama": re.compile(r"\bBBC\s+Panorama\b",re.I),
    "BBC iPlayer/Sounds": re.compile(r"\b(?:BBC\s+)?(?:iPlayer|BBC\s+Sounds)\b",re.I),
    "AJ Inside Story": re.compile(r"\bAl\s*Jazeera(?:['’]s)?\s+Inside\s+Story\b",re.I),
    "AJ Counting the Cost": re.compile(r"\bAl\s*Jazeera(?:['’]s)?\s+Counting\s+the\s+Cost\b",re.I),
    "Guardian Today in Focus": re.compile(r"\bThe\s+Guardian(?:['’]s)?\s+Today\s+in\s+Focus\b",re.I),
    "Guardian Politics Weekly": re.compile(r"\bThe\s+Guardian(?:['’]s)?\s+Politics\s+Weekly\b",re.I),
}
PROGRAM_CONTEXT_CHECKS = {
    "Panorama_media_context": re.compile(r"\b(?:told|speaking (?:to|on)|appeared on|according to|interview(?:ed)? (?:by|with)|on)\s+(?:the\s+)?Panorama\b|\bPanorama\s+(?:programme|program|investigation|interview)\b",re.I),
    "AJ_bare_program_context": re.compile(r"\b(?:told|speaking (?:to|on)|appeared on|according to|interview(?:ed)? (?:by|with)|on)\s+(?:the\s+)?(?:The\s+Take|Fault\s+Lines|Inside\s+Story|Counting\s+the\s+Cost|The\s+Listening\s+Post|The\s+Stream|People\s*&\s*Power|101\s+East)\b",re.I),
    "Guardian_bare_program_context": re.compile(r"\b(?:told|speaking (?:to|on)|appeared on|according to|interview(?:ed)? (?:by|with)|on)\s+(?:the\s+)?(?:Today\s+in\s+Focus|First\s+Edition|Politics\s+Weekly|Full\s+Story|The\s+Long\s+Read)\b",re.I),
}
# Contextual Al Jazeera Sanad service check. Bare "Sanad" is not enough because
# an unrelated Sanad News Agency also appears in the corpus.
PROGRAM_CONTEXT_CHECKS["AJ_Sanad_context"] = re.compile(
    r"(?:\b(?:according to|obtained by|reviewed by|analysed by|analyzed by|verified by|authenticated by)\s+Sanad\b)|"
    r"(?:\bSanad\s+(?:also\s+)?(?:has\s+)?(?:found|verified|analysed|analyzed|documented|authenticated|shows?|concluded|estimates?)\b)|"
    r"(?:\bSanad['’]s\s+(?:analysis|investigation|verification)\b)|"
    r"(?:\b(?:fact-checking|verification)\s+(?:agency|unit|team)\s+Sanad\b)", re.I)
WIRE_HARD_CHECKS = {
    "Reuters": re.compile(r"\bReuters\b",re.I),
    "Associated Press": re.compile(r"\bAssociated\s+Press\b",re.I),
    "Agence France-Presse": re.compile(r"\bAgence\s+France[- ]Presse(?:e)?\b",re.I),
    "bare_AP_attribution": re.compile(r"\b(?:told|according to|speaking to|interview(?:ed)? (?:by|with))\s+(?:the\s+)?AP\b|\b(?:the\s+)?AP\s+(?:reported|reports|said|says|found|wrote|noted|cited|reporter|journalist|photographer|news agency)\b"),
    "bare_AFP_attribution": re.compile(r"\b(?:told|according to|speaking to|interview(?:ed)? (?:by|with))\s+(?:the\s+)?AFP\b|\b(?:the\s+)?AFP\s+(?:reported|reports|said|says|found|wrote|noted|cited|reporter|journalist|photographer|news agency)\b",re.I),
}
ARTIFACT_CHECKS = {
    "recommended_stories": re.compile(r"recommended stories", re.IGNORECASE),
    "list_marker_artifact": re.compile(r"recommended stories\s+list of\s+\d+\s+items?|[-–—]\s*list\s+\d+\s+of\s+\d+", re.IGNORECASE),
    "leading_published": re.compile(r"^\s*-?\s*published\b", re.IGNORECASE),
    "external_annotation": re.compile(r",\s*external\s*,", re.IGNORECASE),
    "newsletter": re.compile(r"\bnewsletter\b", re.IGNORECASE),
    "sign_up": re.compile(r"\bsign up\b", re.IGNORECASE),
    "copyright": re.compile(r"\bcopyright\b|©", re.IGNORECASE),
    "all_rights_reserved": re.compile(r"\ball rights reserved\b", re.IGNORECASE),
    "mojibake_markers": re.compile(r"[âÂÃ]"),
    "told_the_scar": re.compile(r"\btold the (?:that|it|he|she|they|we|there)\b", re.IGNORECASE),
    "target_source_possessive_scar": re.compile(r"\b(?:said|reported|told|speaking in an interview)['’]s\s+(?:(?:Gaza|Business)\s+)?Today\s+(?:programme|program)\b|\b(?:said|reported|told)['’]s\s+Newsnight\b", re.IGNORECASE),
    "orphan_possessive_said": re.compile(r"(?<!\w)['’]s\s+(?:said|reported|noted|added|explained)\b", re.IGNORECASE),
    "orphan_possessive_program": re.compile(r"(?<!\w)['’]s\s+(?:Newsnight|HARDtalk|UpFront|The Take|Fault Lines)\b",re.I),
    "orphan_possessive_person": re.compile(r"(?<!\w)['’]s\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]+){1,4}\b"),
    "target_reporter_internal_sentinel": re.compile(r"__TARGET_REPORTER_ATTR__"),
    "malformed_news_agency_article": re.compile(r"\b(?:a|an|the)\s+the\s+news\s+agency\b|\bthe\s+news\s+agency\s+news\s+agency\b",re.I),
    "synthetic_reporting_reporter": re.compile(r"\bReporting from\s+[^.!?]{1,120},\s*a reporter\s+(?:said|reported|noted|added|explained|wrote|found)\b",re.I),
    "contributed_reporting_credit": re.compile(r"\bcontributed (?:reporting|to this report)\b", re.IGNORECASE),
}
HARD_ARTIFACT_NAMES = {
    "recommended_stories", "list_marker_artifact", "leading_published",
    "external_annotation", "mojibake_markers", "told_the_scar",
    "target_source_possessive_scar", "orphan_possessive_said", "orphan_possessive_program",
    "orphan_possessive_person", "target_reporter_internal_sentinel", "malformed_news_agency_article",
    "synthetic_reporting_reporter", "contributed_reporting_credit",
}
SUSPICIOUS_TFIDF_RE = re.compile(
    r"\b(?:bbc|al jazeera|the guardian|today programme|radio 4|newsnight|hardtalk|upfront|fault lines|"
    r"the take|iplayer|reuters|associated press|agence france|recommended stories|list of items|"
    r"reporter said|reporter reported|reporting from|an the news agency|a the news agency|the the news agency|"
    r"news agency news agency)\b",re.I
)
WORD_RE = re.compile(r"\b\w+(?:['’]\w+)?\b", re.UNICODE)
NAME_TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’.-][^\W\d_]+)*", re.UNICODE)


# FINAL V4 audit overrides after full inspection of the previous local run.
# Wire agencies are external sources, not target labels, so only malformed
# cleaner products are hard failures; genuine Reuters/AP/AFP prose may remain.
WIRE_HARD_CHECKS = {}
PROGRAM_CHECKS.update({
    "BBC Newshour": re.compile(r"\bNewshour(?:\s+programme|\s+program)?\b",re.I),
    "BBC Newsday": re.compile(r"\bNewsday(?:\s+programme|\s+program)?\b",re.I),
    "BBC World Tonight": re.compile(r"\bWorld\s+Tonight(?:\s+programme|\s+program)?\b",re.I),
    "BBC World at One": re.compile(r"\b(?:The\s+)?World\s+[Aa]t\s+One(?:\s+programme|\s+program)?\b",re.I),
    "BBC Woman's Hour": re.compile(r"\bWoman['’]s\s+Hour(?:\s+programme|\s+program)?\b",re.I),
    "BBC Gaza Lifeline": re.compile(r"\bGaza\s+Lifeline(?:\s+programme|\s+program)?\b",re.I),
    "BBC World Business Report": re.compile(r"\bWorld\s+Business\s+Report\b",re.I),
    "BBC Good Morning Scotland": re.compile(r"\bGood\s+Morning\s+Scotland(?:\s+programme|\s+program)?\b",re.I),
    "BBC Scotcast": re.compile(r"\bScotcast\s+podcast\b",re.I),
    "BBC Sunday with Laura Kuenssberg": re.compile(r"\bSunday\s+with\s+Laura\s+Kuenssberg(?:\s+programme|\s+program)?\b",re.I),
    "AJ Mawazine": re.compile(r"\bMawazine\s+(?:programme|program)\b",re.I),
})
ARTIFACT_CHECKS.update({
    "verb_possessive_scar": re.compile(r"\b(?:said|told|reported|wrote|added|noted|explained|found|stated|confirmed|speaking|interview)['’]s\b",re.I),
    "reporters_possessive_scar": re.compile(r"\breporters['’]s\b",re.I),
    "interview_one_program_scar": re.compile(r"\binterview\s+One['’]s\s+Sunday\s+with\s+Laura\s+Kuenssberg\b",re.I),
})
HARD_ARTIFACT_NAMES.update({"verb_possessive_scar","reporters_possessive_scar","interview_one_program_scar"})
SUSPICIOUS_TFIDF_RE = re.compile(
    r"\b(?:bbc|al jazeera|the guardian|today programme|radio 4|newsnight|hardtalk|newshour|newsday|"
    r"world tonight|world at one|woman's hour|gaza lifeline|world business report|good morning scotland|"
    r"scotcast|sunday with laura kuenssberg|upfront|fault lines|the take|mawazine|iplayer|"
    r"recommended stories|list of items|said's|told's|reported's|reporters's|__target_reporter_attr__)\b",re.I)



# FINAL V5 audit additions.
# Wire names are legitimate in article prose, but a trailing "With Reuters / AFP"
# source credit is structural furniture and must not survive. Wire names are also
# restored to the TF-IDF suspicious-feature list so any unexpected portal-specific
# source-credit residue is surfaced for review rather than silently ignored.
_WIRE_CREDIT_NAME_V5 = r"(?:Reuters|Associated\s+Press|Agence\s+France[- ]Presse|AFP)"
ARTIFACT_CHECKS.update({
    'trailing_wire_credit_v5': re.compile(
        rf"(?:^|[.!?]\s+)[-–—]?\s*(?:With|Additional\s+reporting\s+by)\s+{_WIRE_CREDIT_NAME_V5}"
        rf"(?:\s*(?:,|and|&)\s*{_WIRE_CREDIT_NAME_V5})*\s*\.?\s*$", re.I),
})
HARD_ARTIFACT_NAMES.add('trailing_wire_credit_v5')

# FINAL V7 audit additions. The V6 person-like branch was too broad because
# case-insensitive [A-Z] also matched ordinary sentence-initial nouns ("Things")
# and "It" in grammatical impersonal passives. Keep a precise regression check
# for the pronoun patterns actually produced by the former unsafe told->said rule.
ARTIFACT_CHECKS.update({
    'passive_said_pronoun_scar_v7': re.compile(
        r"\b(?:I|we|you|they|he|she)\s+"
        r"(?:was|were|have\s+been|has\s+been|had\s+been)\s+said(?:\s+(?:that|he|she|they|we|you|it|there)\b|\s*,)", re.I),
    'said_reporters_scar_v6': re.compile(r"\bsaid\s+(?:reporters|journalists|correspondents)\b", re.I),
})
HARD_ARTIFACT_NAMES.discard('passive_said_person_scar_v6')
HARD_ARTIFACT_NAMES.update({'passive_said_pronoun_scar_v7','said_reporters_scar_v6'})

# FINAL V9 regressions: catch synthetic scars found in the full V8 review.
ARTIFACT_CHECKS.update({
    'reports_possessive_scar_v9': re.compile(r"\breports['’]s\b", re.I),
    'reports_teams_scar_v9': re.compile(r"\breports\s+teams?\s+on\s+the\s+ground\b", re.I),
    'wire_dangling_its_correspondent_v9': re.compile(
        r"\breports\s+indicated\s+its\s+(?:correspondent|reporter|journalist|photographer)\b", re.I),
    'lowercase_generated_attribution_start_v9': re.compile(
        r"(?:^|[.!?]\s+)(?:according to reports|speaking to reporters|speaking in an interview|"
        r"in an interview|interviewed by reporters|appeared in an interview)\b"),
})
HARD_ARTIFACT_NAMES.update({
    'reports_possessive_scar_v9','reports_teams_scar_v9',
    'wire_dangling_its_correspondent_v9','lowercase_generated_attribution_start_v9',
})

# High-precision detector for the Al Jazeera extraction failure where a related
# media/card headline is flattened immediately before the actual article lead.
_AJ_LEADING_HEADLINE_ACTION_RE = re.compile(
    r"\b(?:kills?|dies?|blows?|destroys?|collapses?|pounds?|waits?|suggests?|reopens?|"
    r"restricts?|grows?|worr(?:y|ies)|deteriorates?|slams?|attacks?|backs?|rejects?|"
    r"approves?|halts?|blocks?|allows?|returns?|warns?|urges?|calls?|faces?|sparks?|"
    r"shakes?|hits?|cuts?|raises?|falls?|rises?)\b", re.I)
_AJ_BODY_START_RE = re.compile(
    r"^(?:A|An|The|At\s+least|Israel|Iran(?:['’]s)?|United\s+States|US|Fourteen|Strong|"
    r"Hundreds|Thousands|Palestinian|Israeli|Qatar|Lebanon|Syria|Gaza)\b")
_AJ_BODY_FINITE_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|will|would|can|could|may|might|must|should|"
    r"says?|said|claimed|claims|appealed|appeals|launched|launches|killed|kills|"
    r"condemned|condemns|warned|warns|called|calls|reported|reports|showed|shows|"
    r"began|remains|faces|announced|approved|rejected|urged|accused|continues|created|"
    r"released|opened|closed|rose|fell|slumped|surged|moved|pushed|sought|took|made|"
    r"found|adding|adds?)\b", re.I)
_AJ_BAD_PREFIX_FINAL_WORDS = {
    "in","on","at","to","from","for","of","with","by","as","into","onto","across",
    "over","under","amid","among","against","after","before","during","through","around",
    "near","via","and","or","but","nor","yet","so","that","which","who","whose","whom",
    "a","an","the","this","these","those","its","their","his","her","our","your","my",
    "one","more","less","than","about","despite","without","within","between","since",
    "central","northern","southern","eastern","western","occupied","besieged","ongoing",
    "following","including","excluding","toward","towards",
}

def _aj_leading_heading(text: str) -> str:
    b=re.sub(r"\s+"," ",text or "").strip()
    words=list(re.finditer(r"\S+",b))
    for n in range(5,min(16,len(words))):
        end=words[n-1].end(); prefix=b[:end]; rest=b[end:].lstrip()
        if re.search(r"[.!?;:]",prefix): continue
        toks=re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’][A-Za-z]+)?",prefix)
        if not toks: continue
        last=toks[-1].casefold().replace("’", "'")
        if last in _AJ_BAD_PREFIX_FINAL_WORDS or last.endswith("ing"): continue
        if not _AJ_BODY_START_RE.match(rest): continue
        first_clause=re.split(r"[.!?;:]",rest,maxsplit=1)[0]
        if not _AJ_BODY_FINITE_RE.search(" ".join(first_clause.split()[:14])): continue
        if (re.match(r"^(?:How|Why|What)\b",prefix,re.I)
            or _AJ_LEADING_HEADLINE_ACTION_RE.search(prefix)
            or re.match(r"^(?:Israel|Iran|United\s+States|US|China|Russia)\s+has\s+\w+(?:ed|en)\b",prefix,re.I)):
            return prefix
    return ""
SUSPICIOUS_TFIDF_RE = re.compile(
    r"\b(?:bbc|al jazeera|the guardian|today programme|radio 4|newsnight|hardtalk|newshour|newsday|"
    r"world tonight|world at one|woman's hour|gaza lifeline|world business report|good morning scotland|"
    r"scotcast|sunday with laura kuenssberg|upfront|fault lines|the take|mawazine|sanad|iplayer|"
    r"reuters|associated press|agence france|recommended stories|list of items|said's|told's|reported's|"
    r"reports's|reporters's|reports teams|reports indicated its correspondent|said reporters|"
    r"__target_reporter_attr__)\b",re.I)

def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); keys.append(k)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def quantiles(vals):
    a=np.asarray(vals,dtype=float)
    return {
        "n":int(len(a)), "mean":float(np.mean(a)), "std":float(np.std(a)),
        "min":float(np.min(a)), "p10":float(np.quantile(a,.10)),
        "p25":float(np.quantile(a,.25)), "median":float(np.quantile(a,.50)),
        "p75":float(np.quantile(a,.75)), "p90":float(np.quantile(a,.90)),
        "max":float(np.max(a)),
    }


def name_token(s): return s.casefold().replace("’", "'")

class AuthorMatcher:
    def __init__(self,names):
        by_first=defaultdict(list)
        for name in names:
            toks=tuple(name_token(m.group(0)) for m in NAME_TOKEN_RE.finditer(name))
            if len(toks)>=2: by_first[toks[0]].append(toks)
        for k in by_first: by_first[k].sort(key=len,reverse=True)
        self.by_first=dict(by_first)
    def find_any(self,text):
        ms=list(NAME_TOKEN_RE.finditer(text)); toks=[name_token(m.group(0)) for m in ms]
        for i,t in enumerate(toks):
            for seq in self.by_first.get(t,()):
                L=len(seq)
                if i+L<=len(toks) and tuple(toks[i:i+L])==seq:
                    return text[ms[i].start():ms[i+L-1].end()]
        return None


def body_length_features(r):
    text=r.get("text_clean") or ""
    words=WORD_RE.findall(text)
    return [math.log1p(len(words)), math.log1p(len(text))]


def headline_length_features(r):
    title=r.get("title_clean") or ""
    tw=WORD_RE.findall(title)
    return [math.log1p(len(tw)), math.log1p(len(title))]


def body_plus_headline_length_features(r):
    return body_length_features(r) + headline_length_features(r)


def structural_features(r):
    # V7: body-only structural shortcut diagnostic. Headline length has its own
    # separate diagnostic and must not leak into this feature set.
    text=r.get("text_clean") or ""
    words=WORD_RE.findall(text); n=max(1,len(text))
    return [
        math.log1p(len(words)), math.log1p(len(text)),
        math.log1p(text.count(".")), math.log1p(text.count(",")),
        math.log1p(text.count("?")+text.count("!")),
        math.log1p(text.count('"')+text.count("“")+text.count("”")),
        sum(ch.isdigit() for ch in text)/n,
        sum(not ch.isascii() for ch in text)/n,
    ]


def balanced_topic_rows(rows, topic, seed=17):
    by=defaultdict(list)
    for r in rows:
        if r["topic"]==topic: by[r["portal"]].append(r)
    if any(p not in by or not by[p] for p in PORTALS): return []
    m=min(len(by[p]) for p in PORTALS); rng=random.Random(seed)
    out=[]
    for p in PORTALS:
        x=by[p][:]; rng.shuffle(x); out.extend(x[:m])
    rng.shuffle(out); return out


def month_balanced_topic_rows(rows, topic, seed=17):
    """Sample equal portal counts inside every year-month for diagnostics.

    This mirrors the temporal control planned for the real experiment, so a
    length-only shortcut score cannot be inflated merely because portals have
    different month distributions.
    """
    by=defaultdict(list)
    for r in rows:
        if r["topic"]==topic:
            by[(r["year_month"],r["portal"])].append(r)
    months=sorted({ym for (ym,p) in by})
    rng=random.Random(seed); out=[]
    for ym in months:
        pools={p:list(by.get((ym,p),[])) for p in PORTALS}
        if any(not pools[p] for p in PORTALS):
            continue
        m=min(len(pools[p]) for p in PORTALS)
        for p in PORTALS:
            rng.shuffle(pools[p]); out.extend(pools[p][:m])
    rng.shuffle(out)
    return out


def cv_numeric(rows, feature_fn):
    if not SKLEARN_OK or len(rows)<60: return None
    X=np.asarray([feature_fn(r) for r in rows],dtype=float)
    y=np.asarray([r["portal"] for r in rows])
    clf=make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=123)
    scores=cross_validate(clf,X,y,cv=cv,scoring={"acc":"accuracy","f1":"f1_macro"},n_jobs=1)
    return {
        "accuracy_mean":float(np.mean(scores["test_acc"])),
        "accuracy_std":float(np.std(scores["test_acc"])),
        "macro_f1_mean":float(np.mean(scores["test_f1"])),
        "macro_f1_std":float(np.std(scores["test_f1"])),
        "n_balanced":len(rows),
    }


def tfidf_features(rows, topic, out_csv: Path, seed=17):
    if not SKLEARN_OK: return None
    br=month_balanced_topic_rows(rows,topic,seed)
    if not br: return None
    # Final main classifier will use body text only; headline is retained for
    # audit/qualitative analysis but deliberately excluded to avoid a strong
    # headline-length shortcut.
    texts=[r["text_clean"] for r in br]; y=[r["portal"] for r in br]
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=4,max_df=.97,max_features=45000,
                        sublinear_tf=True,strip_accents=None,lowercase=True)
    X=vec.fit_transform(texts)
    clf=LinearSVC(class_weight="balanced")
    clf.fit(X,y)
    terms=np.asarray(vec.get_feature_names_out())
    rows_out=[]
    for ci,label in enumerate(clf.classes_):
        idx=np.argsort(clf.coef_[ci])[-50:][::-1]
        for rank,j in enumerate(idx,1):
            rows_out.append({"topic":topic,"portal":label,"rank":rank,"feature":terms[j],"coefficient":float(clf.coef_[ci,j])})
    write_csv(out_csv,rows_out)
    return {"balanced_n":len(br),"vocab_size":len(terms)}



# ==========================================================================
# FINAL V10 AUDIT OVERRIDES
# Catch the residual direct target-brand contexts found in manual V9 review.
# ==========================================================================
OUTLET_CHECKS["Guardian media brand context V10"] = re.compile(
    r"\bGuardian\s+(?:Australia|US|Essential|investigation|interview|readers?|poll|"
    r"newspaper|reporter|correspondent|journalist|columnist|editor)\b|"
    r"\bHarris\s*/\s*Guardian\b",
    re.IGNORECASE,
)
PROGRAM_CHECKS["BBC More from InDepth furniture V10"] = re.compile(
    r"\bMore\s+from\s+InDepth\b", re.IGNORECASE
)
SUSPICIOUS_TFIDF_RE = re.compile(
    rf"(?:{SUSPICIOUS_TFIDF_RE.pattern})|"
    r"(?:\b(?:guardian australia|guardian essential|guardian us|guardian investigation|"
    r"guardian interview|guardian readers|harris guardian|more from indepth)\b)",
    re.IGNORECASE,
)


# ==========================================================================
# FINAL V11 AUDIT OVERRIDES
# Regressions for the V10 full-output review.
# ==========================================================================

# Hyphenated target brand in unambiguous media/source context. Do NOT match the
# unrelated Syrian place-name "Al-Jazeera barracks".
OUTLET_CHECKS["Al-Jazeera media/source context V11"] = re.compile(
    r"(?:\b(?:told|according\s+to|speaking\s+to|interview(?:ed)?\s+(?:by|with)|"
    r"reported\s+by|writing\s+for)\s+(?:the\s+)?Al[-\u2010-\u2015\s]+Jazeera\b)|"
    r"(?:\bAl[-\u2010-\u2015\s]+Jazeera\s+(?:reporters?|journalists?|correspondents?|"
    r"colleague|network|channel|offices?|cameramen?|media)\b)",
    re.I,
)

# High-precision tail metadata that must not survive in model text.
ARTIFACT_CHECKS.update({
    "tail_additional_reporting_v11": re.compile(
        r"(?:^|[.!?]\s+)Additional\s+reporting\s+(?:by|from)\b", re.I),
    "tail_additional_research_v11": re.compile(
        r"(?:^|[.!?]\s+)Additional\s+research\s+by\b", re.I),
    "tail_graphics_credit_v11": re.compile(
        r"(?:^|[.!?]\s+)(?:Graphics\s+by|Visual\s+journalism\s+by|"
        r"(?:Photo|Photography|Photographs)\s+by)\b", re.I),
    "tail_correction_note_v11": re.compile(
        r"(?:^|[.!?]\s+)Correction(?:\s+(?:\d{1,2}\s+[A-Z][a-z]+(?:\s+\d{4})?|"
        r"[A-Z][a-z]+\s+\d{1,2}(?:,\s*\d{4})?))?\s*:", re.I),
    "tail_clarification_note_v11": re.compile(
        r"(?:^|[.!?]\s+)Clarification(?:\s+(?:\d{1,2}\s+[A-Z][a-z]+(?:\s+\d{4})?|"
        r"[A-Z][a-z]+\s+\d{1,2}(?:,\s*\d{4})?))?\s*:", re.I),
    "tail_update_note_v11": re.compile(
        r"(?:^|[.!?]\s+)Update(?:,\s*|\s+)?(?:\d{1,2}\s+[A-Z][a-z]+(?:\s+\d{4})?|"
        r"[A-Z][a-z]+\s+\d{1,2}(?:,\s*\d{4})?)?\s*:", re.I),
    "tail_finding_note_v11": re.compile(
        r"(?:^|[.!?]\s+)Finding\s+\d{1,2}\s+[A-Z][a-z]+(?:\s+\d{4})?\s*:", re.I),
    "tail_collaboration_footer_v11": re.compile(
        r"(?:^|[.!?]\s+)(?:This\s+(?:article|piece|story)\s+(?:is|was)\s+"
        r"(?:published\s+in\s+collaboration\s+with|co-published\s+with|"
        r"produced\s+for|produced\s+in\s+partnership\s+with)|"
        r"An\s+earlier\s+version\s+of\s+this\s+article\s+appeared\s+in)\b", re.I),
    "tail_with_agencies_v11": re.compile(
        r"(?:^|[.!?]\s+)With\s+agencies\s*\.?\s*$", re.I),
    "tail_agency_contributed_story_v11": re.compile(
        r"(?:^|[.!?]\s+)[^.!?]{0,150}\b(?:Reuters|Associated\s+Press|AP|AFP|"
        r"Agence\s+France[- ]Presse)\b[^.!?]{0,120}\bcontributed\s+to\s+this\s+story\b", re.I),
    "duplicate_said_v11": re.compile(r"\bsaid\s+said\b", re.I),
})
HARD_ARTIFACT_NAMES.update({
    "tail_additional_reporting_v11","tail_additional_research_v11",
    "tail_graphics_credit_v11","tail_correction_note_v11",
    "tail_clarification_note_v11","tail_update_note_v11",
    "tail_finding_note_v11","tail_collaboration_footer_v11",
    "tail_with_agencies_v11","tail_agency_contributed_story_v11",
    "duplicate_said_v11",
})

SUSPICIOUS_TFIDF_RE = re.compile(
    rf"(?:{SUSPICIOUS_TFIDF_RE.pattern})|"
    r"(?:\b(?:additional reporting|additional research|graphics by|visual journalism|"
    r"published in collaboration|co-published|with agencies|al-jazeera)\b)",
    re.IGNORECASE,
)



# ==========================================================================
# FINAL V12 AUDIT ADDITIONS
# Mechanical regressions found only after inspection of the complete V11 model
# text. These are high-confidence hard failures, not subjective style checks.
# ==========================================================================

# "A Jazeera" is an observed typo-variant of the target outlet in source
# attribution grammar. Do not flag unrelated ordinary words; require a source
# context.
OUTLET_CHECKS["A Jazeera attribution typo V12"] = re.compile(
    r"\b(?:told|according\s+to|speaking\s+to|interviewed\s+by|interview\s+with)"
    r"\s+(?:the\s+)?A\s+Jazeera\b",
    re.IGNORECASE,
)

ARTIFACT_CHECKS.update({
    "double_comma_v12": re.compile(r",\s*,"),
    "comma_colon_v12": re.compile(r",\s*:"),
    "exact_double_period_v12": re.compile(r"(?<!\.)\.\.(?!\.)"),
    "reporting_from_comma_colon_v12": re.compile(
        r"\bReporting\s+from\s+[^.!?]{1,120},\s*:", re.IGNORECASE),
    "reporting_from_double_comma_v12": re.compile(
        r"\bReporting\s+from\s+[^.!?]{1,120},\s*,", re.IGNORECASE),
    "double_comma_reporting_from_v12": re.compile(
        r",\s*,\s*reporting\s+from\s+[^.!?]{1,120}\.", re.IGNORECASE),
    "double_comma_adding_that_v12": re.compile(
        r",\s*,\s*adding\s+that\b", re.IGNORECASE),
    "reporting_verb_parenthetical_tag_v12": re.compile(
        r"\b(?:said|says|wrote|reported|reports|noted|added|stated|confirmed|found)"
        r"\s+\((?:AP|AFP|PDF|ICRC)\)", re.IGNORECASE),
    "bare_pdf_link_annotation_v12": re.compile(r"\(\s*PDF\s*\)", re.IGNORECASE),
    "told_news_agency_sentence_break_v12": re.compile(
        r"\btold\s+the\s+news\s+agency,\s+(?=[A-Z])", re.IGNORECASE),
})
HARD_ARTIFACT_NAMES.update({
    "double_comma_v12","comma_colon_v12","exact_double_period_v12",
    "reporting_from_comma_colon_v12","reporting_from_double_comma_v12",
    "double_comma_reporting_from_v12","double_comma_adding_that_v12",
    "reporting_verb_parenthetical_tag_v12","bare_pdf_link_annotation_v12",
    "told_news_agency_sentence_break_v12",
})

SUSPICIOUS_TFIDF_RE = re.compile(
    rf"(?:{SUSPICIOUS_TFIDF_RE.pattern})|"
    r"(?:\ba\s+jazeera\b|\bsaid\s+(?:ap|afp)\b|\bpdf\b)",
    re.IGNORECASE,
)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="data/corpus_clean_dedup_candidate.jsonl")
    ap.add_argument("--audit",default="data/corpus_clean_dedup_audit_candidate.jsonl")
    ap.add_argument("--author-list",default="outputs/corpus_build/global_author_names.txt")
    ap.add_argument("--out-dir",default="outputs/corpus_audit")
    ap.add_argument("--sample-per-cell",type=int,default=25)
    ap.add_argument("--seed",type=int,default=20260817)
    args=ap.parse_args()

    model=read_jsonl(Path(args.model)); audit=read_jsonl(Path(args.audit)); audit_by={r["article_id"]:r for r in audit}
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    print(f"Auditing {len(model):,} articles")

    # Model-input policy: the main classifier must receive BODY ONLY. This
    # catches accidental headline concatenation before any split/training.
    model_input_policy_violations = sum(
        1 for r in model if (r.get("model_text") or "") != (r.get("text_clean") or "")
    )

    # V8 extraction-integrity policy: every surviving BBC.com article must have
    # passed the conservative exact-title + valid-body-start rule.
    bbc_com_heading_policy_violations = []
    for r in model:
        a = audit_by.get(r["article_id"], {})
        u = a.get("final_url") or a.get("url") or ""
        if r.get("portal") == "bbc" and re.search(r"https?://(?:www\.)?bbc\.com/", u, re.I):
            method = a.get("leading_extraction_heading_method") or ""
            if method != "bbc_com_exact_metadata_h1_validated":
                bbc_com_heading_policy_violations.append({
                    "article_id": r["article_id"], "portal": r["portal"], "topic": r["topic"],
                    "title": r.get("title_clean"), "url": u, "method": method,
                })
    write_csv(out/"bbc_com_heading_policy_violations.csv", bbc_com_heading_policy_violations)


    # V9: no surviving Al Jazeera model text may match the high-confidence
    # embedded-media-heading failure mode corrected at extraction integrity.
    aljazeera_leading_heading_policy_violations=[]
    for r in model:
        if r.get("portal") != "aljazeera":
            continue
        h=_aj_leading_heading(r.get("text_clean") or "")
        if h:
            aljazeera_leading_heading_policy_violations.append({
                "article_id":r["article_id"],"portal":r["portal"],"topic":r["topic"],
                "title":r.get("title_clean"),"suspected_heading":h,
                "text_prefix":(r.get("text_clean") or "")[:260],
            })
    write_csv(out/"aljazeera_leading_heading_policy_violations.csv",
              aljazeera_leading_heading_policy_violations)

    # Portal-blind regression check for malformed leading fragments. After
    # optional opening quotes/brackets, ordinary written news prose should not
    # begin with a lowercase alphabetic fragment.
    body_start_integrity_violations=[]
    for r in model:
        txt=(r.get("text_clean") or "").lstrip()
        probe=txt.lstrip("\"'“”‘’([{")
        first_alpha=next((ch for ch in probe if ch.isalpha()), "")
        if first_alpha and first_alpha.islower():
            body_start_integrity_violations.append({
                "article_id":r["article_id"],"portal":r["portal"],"topic":r["topic"],
                "title":r.get("title_clean"),"text_prefix":txt[:220],
            })
    write_csv(out/"body_start_integrity_violations.csv",body_start_integrity_violations)

    # Direct leakage and suspicious artifact counts by portal/topic.
    leakage_rows=[]; hard_total=(model_input_policy_violations +
                                len(bbc_com_heading_policy_violations) +
                                len(aljazeera_leading_heading_policy_violations) +
                                len(body_start_integrity_violations))
    for r in model:
        txt=(r.get("title_clean") or "")+"\n"+(r.get("text_clean") or "")
        for kind,checks in [("OUTLET_ID",OUTLET_CHECKS),("TARGET_PROGRAM",PROGRAM_CHECKS),("TARGET_PROGRAM_CONTEXT",PROGRAM_CONTEXT_CHECKS),("WIRE_ID",WIRE_HARD_CHECKS)]:
            for name,pat in checks.items():
                if pat.search(txt):
                    leakage_rows.append({"article_id":r["article_id"],"portal":r["portal"],"topic":r["topic"],"severity":"HARD","kind":kind,"pattern":name,"title":r["title_clean"]})
                    hard_total+=1
        for name,pat in ARTIFACT_CHECKS.items():
            if pat.search(txt):
                severity = "HARD" if name in HARD_ARTIFACT_NAMES else "REVIEW"
                leakage_rows.append({"article_id":r["article_id"],"portal":r["portal"],"topic":r["topic"],"severity":severity,"kind":"ARTIFACT","pattern":name,"title":r["title_clean"]})
                if severity == "HARD": hard_total += 1
    write_csv(out/"residual_leakage_hits.csv",leakage_rows)

    # Structural byline/credit audit. A journalist name may legitimately be
    # article content, so names are HARD only when the current article's author
    # survives in a recognizable byline/credit context.
    author_hits=[]
    for r in model:
        a=audit_by.get(r["article_id"],{})
        raw=a.get("author_raw") or ""
        names=[]
        for chunk in re.split(r"\s*;\s*|\s+and\s+",raw):
            chunk=chunk.strip(" ,|-")
            if len(chunk.split())>=2: names.append(chunk)
        txt=(r.get("title_clean") or "")+"\n"+(r.get("text_clean") or "")
        for name in names:
            esc=re.escape(name).replace(r"\ ",r"\s+")
            pat=re.compile(rf"(?:^|\n)\s*(?:By|Reporting by|Written by)\s+{esc}\b|\b(?:additional reporting by|reporting by|written by)\s+{esc}\b",re.I)
            if pat.search(txt):
                author_hits.append({"article_id":r["article_id"],"portal":r["portal"],"topic":r["topic"],"matched_author":name,"title":r["title_clean"]})
                break
    write_csv(out/"residual_author_hits.csv",author_hits)
    hard_total += len(author_hits)

    # Length / cleaning-impact statistics.
    length_rows=[]; grouped=defaultdict(list)
    for r in model:
        a=audit_by.get(r["article_id"],{})
        wc=a.get("clean_body_words") or len(WORD_RE.findall(r["text_clean"]))
        tc=a.get("clean_title_words") or len(WORD_RE.findall(r["title_clean"]))
        cc=a.get("clean_chars") or len(r["text_clean"])
        frac=a.get("body_fraction_chars_removed")
        grouped[(r["portal"],r["topic"])].append((wc,tc,cc,frac))
    length_summary={}
    for (p,t),vals in sorted(grouped.items()):
        w=[v[0] for v in vals]; tw=[v[1] for v in vals]; ch=[v[2] for v in vals]
        fr=[v[3] for v in vals if isinstance(v[3],(int,float))]
        stats={"body_words":quantiles(w),"title_words":quantiles(tw),"body_chars":quantiles(ch)}
        if fr: stats["fraction_raw_chars_removed"]=quantiles(fr)
        length_summary[f"{p}/{t}"]=stats
        length_rows.append({"portal":p,"topic":t,"n":len(vals),"body_words_mean":np.mean(w),"body_words_median":np.median(w),"body_words_p10":np.quantile(w,.1),"body_words_p90":np.quantile(w,.9),"title_words_mean":np.mean(tw),"chars_removed_fraction_mean":np.mean(fr) if fr else None})
    write_csv(out/"length_and_cleaning_summary.csv",length_rows)

    # Projected size after strict month-wise portal balancing (before 80/10/10 split).
    month_counts=Counter((r["topic"],r["year_month"],r["portal"]) for r in model)
    projection={}; monthly_rows=[]
    for topic in sorted({r["topic"] for r in model}):
        total=0
        months=sorted({r["year_month"] for r in model if r["topic"]==topic})
        for ym in months:
            cs={p:month_counts[(topic,ym,p)] for p in PORTALS}
            m=min(cs.values())
            total += 3*m
            monthly_rows.append({"topic":topic,"year_month":ym,**cs,"minimum_per_portal":m,"balanced_total":3*m})
        projection[topic]=total
    write_csv(out/"month_balance_projection.csv",monthly_rows)

    # Random human-quality sample from every portal/topic cell.
    rng=random.Random(args.seed); sample=[]
    bycell=defaultdict(list)
    for r in model: bycell[(r["portal"],r["topic"])].append(r)
    for key,pool in sorted(bycell.items()):
        x=pool[:]; rng.shuffle(x)
        for r in x[:args.sample_per_cell]:
            a=audit_by.get(r["article_id"],{})
            sample.append({"article_id":r["article_id"],"portal":r["portal"],"topic":r["topic"],"year_month":r["year_month"],"title":r["title_clean"],"text_prefix":r["text_clean"][:650],"url":a.get("final_url") or a.get("url"),"topic_relevance_rule":a.get("topic_relevance_rule")})
    write_csv(out/"manual_quality_sample.csv",sample)
    # Secondary QA sample is disjoint from the primary sample: 420 unique rows.
    primary_ids={r["article_id"] for r in sample}
    rng2=random.Random(args.seed+991); sample2=[]
    for key,pool in sorted(bycell.items()):
        x=[r for r in pool if r["article_id"] not in primary_ids]; rng2.shuffle(x)
        for r in x[:20]:
            a=audit_by.get(r["article_id"],{})
            sample2.append({"article_id":r["article_id"],"portal":r["portal"],"topic":r["topic"],"year_month":r["year_month"],"title":r["title_clean"],"text_prefix":r["text_clean"][:650],"url":a.get("final_url") or a.get("url"),"topic_relevance_rule":a.get("topic_relevance_rule")})
    write_csv(out/"manual_quality_sample_secondary.csv",sample2)

    # Shortcut diagnostics.
    body_length_cv={}; headline_length_cv={}; combined_length_cv={}; structural_cv={}; tfidf_meta={}
    for topic in sorted({r["topic"] for r in model}):
        br=month_balanced_topic_rows(model,topic,args.seed)
        body_length_cv[topic]=cv_numeric(br,body_length_features)
        headline_length_cv[topic]=cv_numeric(br,headline_length_features)
        combined_length_cv[topic]=cv_numeric(br,body_plus_headline_length_features)
        structural_cv[topic]=cv_numeric(br,structural_features)
        tfidf_meta[topic]=tfidf_features(model,topic,out/f"tfidf_top_features_{topic}.csv",args.seed)

    suspicious=[]
    for topic in sorted({r["topic"] for r in model}):
        fp=out/f"tfidf_top_features_{topic}.csv"
        if fp.exists():
            with fp.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if SUSPICIOUS_TFIDF_RE.search(row.get("feature") or ""):
                        suspicious.append(row)
    write_csv(out/"tfidf_suspicious_features.csv",suspicious)

    neutral_rows=[]
    for p in PORTALS:
        for t in sorted({r["topic"] for r in model}):
            rr=[r for r in model if r["portal"]==p and r["topic"]==t]
            counts=[]
            for r in rr:
                a=audit_by.get(r["article_id"],{})
                st=a.get("body_cleaning_stats") or {}
                counts.append(int(st.get("target_attributions_normalized",0) or 0)+int(st.get("target_source_scars_repaired_v4",0) or 0))
            n=len(rr); hits=sum(c>0 for c in counts)
            neutral_rows.append({"portal":p,"topic":t,"n":n,
                "target_source_normalized_articles":hits,
                "target_source_normalized_fraction":hits/max(1,n),
                "target_source_normalizations_total":sum(counts),
                "target_source_normalizations_mean":sum(counts)/max(1,n)})
    write_csv(out/"neutralization_phrase_distribution.csv",neutral_rows)

    # Aggregate residual artifact counts.
    residual_counts=Counter((x["kind"],x["pattern"],x["portal"],x["topic"]) for x in leakage_rows)
    residual_table=[{"kind":k,"pattern":pat,"portal":p,"topic":t,"articles":n} for (k,pat,p,t),n in sorted(residual_counts.items())]
    write_csv(out/"residual_counts_by_cell.csv",residual_table)

    summary={
        "n_articles":len(model),
        "counts_by_portal_topic":{f"{p}/{t}":n for (p,t),n in sorted(Counter((r["portal"],r["topic"]) for r in model).items())},
        "hard_residual_leakage_hits":hard_total,
        "model_text_body_only_policy_violations":model_input_policy_violations,
        "bbc_com_heading_policy_violations":len(bbc_com_heading_policy_violations),
        "aljazeera_leading_heading_policy_violations":len(aljazeera_leading_heading_policy_violations),
        "body_start_integrity_violations":len(body_start_integrity_violations),
        "review_flag_rows":sum(1 for x in leakage_rows if x.get("severity") == "REVIEW"),
        "review_flag_articles":len({x["article_id"] for x in leakage_rows if x.get("severity") == "REVIEW"}),
        "residual_author_hits":len(author_hits),
        "month_balanced_projection_before_split":projection,
        "body_length_only_month_balanced_5fold_diagnostic":body_length_cv,
        "headline_length_only_month_balanced_5fold_diagnostic":headline_length_cv,
        "body_plus_headline_length_month_balanced_5fold_diagnostic":combined_length_cv,
        "structure_only_month_balanced_5fold_diagnostic":structural_cv,
        "tfidf_diagnostic":tfidf_meta,
        "tfidf_suspicious_top_feature_rows":len(suspicious),
        "model_sha256":sha256_file(Path(args.model)),
        "audit_sha256":sha256_file(Path(args.audit)),
        "manual_quality_primary_n":len(sample),
        "manual_quality_secondary_n":len(sample2),
        "note":"Chance accuracy for 3 balanced portal classes is 0.333. Numeric diagnostics, the body-only structural diagnostic, and body-only TF-IDF are month-balanced PRE-SPLIT leakage checks, not final reported model results.",
    }
    (out/"audit_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    print("\n=== AUDIT SUMMARY ===")
    print(json.dumps(summary,indent=2))
    print("\nDo NOT create final splits until manual_quality_sample.csv and TF-IDF top features have been inspected.")

if __name__=="__main__":
    main()
