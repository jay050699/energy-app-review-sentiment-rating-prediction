"""Collect public English-language energy-app reviews from Google Play and Apple.

Review collection is cached under data/raw/collection_staging. The accepted English
reviews can be promoted to the existing data/raw/energy_app_reviews_raw.csv file
using --promote after reviewing the staging report.
"""

import argparse
import csv
import hashlib
import html
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from google_play_scraper import Sort, app as google_app, reviews as google_reviews
from langdetect import DetectorFactory, LangDetectException, detect_langs

DetectorFactory.seed = 0

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "energy_apps_shortlist.csv"
STAGING = ROOT / "data" / "raw" / "collection_staging"
GOOGLE_DIR = STAGING / "google"
APPLE_DIR = STAGING / "apple"
SUMMARY_DIR = STAGING / "summary"
FINAL_RAW = ROOT / "data" / "raw" / "energy_app_reviews_raw.csv"
FINAL_DATASET_SUMMARY = ROOT / "reports" / "final_dataset_summary.csv"

RAW_FIELDS = [
    "platform", "app_name", "app_id", "category", "country", "review_id",
    "review_text", "rating", "review_date", "thumbs_up_count", "app_version",
    "review_title", "source_url"
]
CLEAN_FIELDS = [
    "app_name", "app_id", "category", "country", "review_text", "rating",
    "review_date", "thumbs_up_count", "app_version", "word_count", "char_count",
    "platform", "review_id", "review_title", "source_url", "language", "language_probability"
]


class AccessStopped(Exception):
    """A store responded with a rate/access restriction; do not try to evade it."""


def make_key(platform, app_id, remote_id):
    raw = f"{platform}:{app_id}:{remote_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def clean_display_text(s):
    s = html.unescape(str(s or ""))
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"https?://\S+|www\.\S+", " [URL] ", s)
    s = re.sub(r"\b[\w.%+\-]+@[\w.\-]+\.[A-Za-z]{2,}\b", " [EMAIL] ", s)
    # Avoid collecting obvious long phone-number-like strings in publicly authored text.
    s = re.sub(r"(?<!\w)\+?\d[\d\s().-]{8,}\d(?!\w)", " [PHONE] ", s)
    return re.sub(r"\s+", " ", s).strip()


def row_key(row):
    return (str(row["platform"]), str(row["app_id"]), str(row["review_id"]))


def load_rows(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except Exception as exc:
        print(f"  WARNING: Could not load cache {path}: {exc}")
        return []


def append_new_rows(path, rows, keys):
    fresh = [r for r in rows if row_key(r) not in keys]
    if not fresh:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    has_header = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS, extrasaction="ignore")
        if not has_header:
            writer.writeheader()
        writer.writerows(fresh)
    keys.update(row_key(r) for r in fresh)
    return len(fresh)


def normalized(text):
    return re.sub(r"\s+", " ", text.casefold()).strip()


def english_probability(text, threshold, min_words):
    if len(re.findall(r"\b\w+\b", text)) < min_words or len(text) < 15:
        return None, "too_short"
    try:
        guesses = detect_langs(text)
        if guesses and guesses[0].lang == "en" and guesses[0].prob >= threshold:
            return float(guesses[0].prob), "english"
        return float(guesses[0].prob) if guesses else None, "not_confident_english"
    except LangDetectException:
        return None, "undetermined_language"


def google_record(app_cfg, r, country):
    remote_id = str(r.get("reviewId") or "")
    body = clean_display_text(r.get("content"))
    if not remote_id or not body:
        return None
    stamp = r.get("at")
    if hasattr(stamp, "isoformat"):
        stamp = stamp.isoformat()
    return {
        "platform": "google_play", "app_name": app_cfg["app_name"],
        "app_id": app_cfg["google_app_id"], "category": app_cfg["category"],
        "country": country, "review_id": make_key("google_play", app_cfg["google_app_id"], remote_id),
        "review_text": body, "rating": r.get("score"), "review_date": stamp,
        "thumbs_up_count": r.get("thumbsUpCount"),
        "app_version": r.get("reviewCreatedVersion"), "review_title": "",
        "source_url": f"https://play.google.com/store/apps/details?id={app_cfg['google_app_id']}",
    }


def apple_review_body(entry):
    content = entry.get("content", {})
    if isinstance(content, list):
        parts = [c for c in content if isinstance(c, dict) and c.get("attributes", {}).get("type") == "text"]
        content = parts[0] if parts else (content[0] if content else {})
    if isinstance(content, dict):
        return content.get("label", "")
    return ""


def apple_record(app_cfg, entry, country):
    if "im:rating" not in entry:
        return None
    remote_id = entry.get("id", {}).get("label", "")
    if not remote_id:
        return None
    body = clean_display_text(apple_review_body(entry))
    if not body:
        return None
    return {
        "platform": "apple_app_store", "app_name": app_cfg["app_name"],
        "app_id": str(app_cfg["apple_app_id"]), "category": app_cfg["category"],
        "country": country,
        "review_id": make_key("apple_app_store", app_cfg["apple_app_id"], remote_id),
        "review_text": body,
        "rating": entry.get("im:rating", {}).get("label", ""),
        "review_date": entry.get("updated", {}).get("label", ""),
        "thumbs_up_count": "", "app_version": entry.get("im:version", {}).get("label", ""),
        "review_title": clean_display_text(entry.get("title", {}).get("label", "")),
        "source_url": f"https://apps.apple.com/{country}/app/id{app_cfg['apple_app_id']}",
    }


def check_google_metadata(app_cfg, country):
    result = google_app(app_cfg["google_app_id"], lang="en", country=country)
    return (str(result.get("title", "")), str(result.get("developer", "")))


def collect_google(app_cfg, country, limit, delay, batch_size):
    safe_id = re.sub(r"[^a-zA-Z0-9]+", "_", app_cfg["google_app_id"])
    path = GOOGLE_DIR / f"{safe_id}__{country}.csv"
    done = path.with_suffix(".done")
    if done.exists():
        try:
            marker=json.loads(done.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            marker={}
        if marker.get("exhausted") or int(marker.get("requested_limit", 0)) >= limit:
            rows=load_rows(path)
            print(f"  Google {country}: cached complete ({len(rows)} rows)")
            return {"platform":"google_play", "app_name":app_cfg["app_name"], "country":country,
                    "status":"cached", "retrieved":len(rows), "error":""}
    rows = load_rows(path)
    keys = {row_key(r) for r in rows}
    token = None
    fetched = 0
    batches = 0
    state = "completed"
    reason = ""
    exhausted = False
    try:
        while fetched < limit:
            count = min(batch_size, limit-fetched)
            page, token = google_reviews(
                app_cfg["google_app_id"], lang="en", country=country,
                sort=Sort.NEWEST, count=count, continuation_token=token,
            )
            batches += 1
            fetched += len(page)
            parsed = [google_record(app_cfg, p, country) for p in page]
            new = append_new_rows(path, [r for r in parsed if r is not None], keys)
            if batches % 5 == 0 or not page or token is None:
                print(f"    {app_cfg['app_name']} / {country}: fetched={fetched}, new_to_cache={new}, cached_total={len(keys)}")
            if not page or token is None:
                exhausted=True
                break
            time.sleep(delay)
        # A larger requested limit can be run later without losing already cached reviews.
        done.write_text(json.dumps({"requested_limit":limit,"exhausted":exhausted,
                                    "completed_utc":datetime.now(timezone.utc).isoformat()}),
                        encoding="utf-8")
    except Exception as exc:
        state = "incomplete"
        reason = f"{type(exc).__name__}: {str(exc)[:200]}"
        print(f"  Google {country} incomplete: {reason}")
    return {"platform":"google_play", "app_name":app_cfg["app_name"], "country":country,
            "status":state, "retrieved":len(keys), "error":reason}


def apple_lookup(session, app_id, country):
    resp = session.get("https://itunes.apple.com/lookup", params={"id":app_id,"country":country}, timeout=25)
    if resp.status_code in (403,429):
        raise AccessStopped(f"Apple lookup HTTP {resp.status_code}")
    resp.raise_for_status()
    return any(str(x.get("trackId", "")) == str(app_id) for x in resp.json().get("results",[]))


def collect_apple(app_cfg, country, max_pages, delay, session):
    app_id = str(app_cfg["apple_app_id"]).strip()
    path = APPLE_DIR / f"{app_id}__{country}.csv"
    done = path.with_suffix(".done")
    if done.exists():
        rows = load_rows(path)
        print(f"  Apple {country}: cached complete ({len(rows)} rows)")
        return {"platform":"apple_app_store","app_name":app_cfg["app_name"],"country":country,
                "status":"cached","retrieved":len(rows), "error":""}
    rows = load_rows(path)
    keys = {row_key(r) for r in rows}
    try:
        if not apple_lookup(session, app_id, country):
            done.write_text("not_available_in_storefront", encoding="utf-8")
            return {"platform":"apple_app_store","app_name":app_cfg["app_name"],"country":country,
                    "status":"not_listed","retrieved":0, "error":""}
        time.sleep(delay)
        errors=[]
        for page in range(1, max_pages+1):
            url = f"https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json"
            try:
                resp = session.get(url, timeout=30)
                if resp.status_code in (403,429):
                    raise AccessStopped(f"Apple reviews HTTP {resp.status_code} on {app_id}/{country}")
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                entries = resp.json().get("feed", {}).get("entry", [])
                if isinstance(entries, dict):
                    entries = [entries]
                parsed = [apple_record(app_cfg, e, country) for e in entries]
                append_new_rows(path, [x for x in parsed if x is not None], keys)
            except AccessStopped:
                raise
            except Exception as exc:
                errors.append(f"page {page}: {type(exc).__name__}")
            time.sleep(delay)
        if errors:
            state="incomplete"
            error="; ".join(errors[:4])
        elif not keys:
            # Apple's public feed sometimes responds with empty pages despite reviews.
            # Leave this job retryable on a later run.
            state="empty_feed"
            error="All tested pages were empty; retry may return different results"
        else:
            done.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
            state="completed"
            error=""
        print(f"  Apple {country}: {len(keys)} cached written reviews ({state})")
        return {"platform":"apple_app_store","app_name":app_cfg["app_name"],"country":country,
                "status":state,"retrieved":len(keys), "error":error}
    except AccessStopped:
        raise
    except Exception as exc:
        return {"platform":"apple_app_store","app_name":app_cfg["app_name"],"country":country,
                "status":"incomplete","retrieved":len(keys), "error":f"{type(exc).__name__}: {exc}"}


def finalize(threshold, min_words, attempts):
    raw_paths = list(GOOGLE_DIR.glob("*.csv")) + list(APPLE_DIR.glob("*.csv"))
    print("\nConsolidating", len(raw_paths), "cached storefront files ...")
    raw_records = []
    for p in sorted(raw_paths):
        raw_records.extend(load_rows(p))
    print("Cached reviews before deduplication:", len(raw_records))
    seen_ids=set()
    unique=[]
    for r in raw_records:
        key=row_key(r)
        if key not in seen_ids:
            seen_ids.add(key)
            unique.append(r)
    print("Unique review IDs across storefronts:", len(unique))

    accepted=[]
    reject_rows=[]
    reasons=Counter()
    duplicate_text=set()
    for r in unique:
        review_text=clean_display_text(r.get("review_text"))
        try:
            rating=int(float(r.get("rating")))
        except (TypeError, ValueError):
            reasons["invalid_rating"]+=1
            continue
        if rating not in (1,2,3,4,5):
            reasons["invalid_rating"]+=1
            continue
        probability, reason = english_probability(review_text, threshold, min_words)
        if reason != "english":
            reasons[reason]+=1
            if len(reject_rows)<100:
                reject_rows.append({"app_name":r.get("app_name"),"platform":r.get("platform"),
                                    "reason":reason,"review_text":review_text[:400]})
            continue
        # Prevent identical text/rating repeats from leaking between train and test.
        # Different rating for same text remains distinct; check manually later.
        text_key=(r.get("app_name"), normalized(review_text), rating)
        if text_key in duplicate_text:
            reasons["duplicate_text_and_rating"]+=1
            continue
        duplicate_text.add(text_key)
        row={k:r.get(k, "") for k in RAW_FIELDS}
        row["review_text"]=review_text
        row["rating"]=rating
        row["word_count"]=len(review_text.split())
        row["char_count"]=len(review_text)
        row["language"]="en"
        row["language_probability"]=round(probability, 5)
        accepted.append(row)

    SUMMARY_DIR.mkdir(parents=True,exist_ok=True)
    ordered=pd.DataFrame(accepted,columns=CLEAN_FIELDS)
    ordered.to_csv(STAGING / "energy_app_reviews_english.csv", index=False,encoding="utf-8")
    pd.DataFrame(attempts, columns=["platform","app_name","country","status","retrieved","error"]).to_csv(SUMMARY_DIR/"storefront_attempts.csv",index=False)
    if not ordered.empty:
        ordered.groupby(["platform","app_name"]).size().rename("usable_english_reviews").reset_index().sort_values("usable_english_reviews",ascending=False).to_csv(SUMMARY_DIR/"reviews_by_app_platform.csv",index=False)
        ordered.groupby("platform").size().rename("usable_english_reviews").to_csv(SUMMARY_DIR/"reviews_by_platform.csv")
        ordered.groupby("rating").size().rename("count").to_csv(SUMMARY_DIR/"rating_distribution.csv")
        ordered.sample(min(70,len(ordered)),random_state=42)[["app_name","platform","country","rating","review_text"]].to_csv(SUMMARY_DIR/"english_sample_for_manual_review.csv",index=False)
    pd.DataFrame(reject_rows).to_csv(SUMMARY_DIR/"rejected_sample_for_manual_review.csv",index=False)
    report={
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "cached_records_before_id_deduplication":len(raw_records),
        "unique_review_ids":len(unique),
        "english_usable_reviews":len(ordered),
        "rejection_reasons":dict(reasons),
        "language_min_words":min_words,
        "language_threshold":threshold,
        "apps_with_english_reviews":int(ordered.app_name.nunique()) if not ordered.empty else 0,
        "storefront_jobs":len(attempts),
        "incomplete_jobs":sum(x["status"]=="incomplete" for x in attempts),
    }
    (SUMMARY_DIR/"collection_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("\n"+"="*68)
    print("FINAL ENGLISH DATASET CANDIDATE COUNT:",len(ordered))
    print("Distinct apps represented:",report["apps_with_english_reviews"])
    print("Incomplete storefront jobs:",report["incomplete_jobs"])
    print("Rejected rows by reason:",report["rejection_reasons"])
    print("Candidate CSV:",STAGING/"energy_app_reviews_english.csv")
    print("Detailed report:",SUMMARY_DIR/"collection_report.json")
    print("Review a sample before declaring the dataset final.")
    return report


def promote_candidate(report):
    """Promote validated English reviews into the project's original raw dataset.

    This deliberately never runs unless --promote is explicitly requested.
    """
    if report["incomplete_jobs"]:
        print("NOT promoting: incomplete storefront jobs remain in this run.")
        return False

    candidate = STAGING / "energy_app_reviews_english.csv"
    data = pd.read_csv(candidate, low_memory=False)
    if data.empty or not {"app_name", "rating", "review_text", "platform", "review_id"}.issubset(data.columns):
        raise ValueError("The staged candidate is empty or missing required columns")

    # Guard against accidentally overwriting the model dataset with fewer reviews
    # due to a temporarily incomplete/filtered collection.
    if FINAL_RAW.exists():
        old_count = sum(1 for _ in FINAL_RAW.open("r", encoding="utf-8")) - 1
        if len(data) < old_count:
            print(f"NOT promoting: candidate contains {len(data):,} reviews, existing raw file has {old_count:,}.")
            return False

    FINAL_RAW.parent.mkdir(parents=True, exist_ok=True)
    FINAL_DATASET_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(FINAL_RAW, index=False, encoding="utf-8")
    summary = (data.groupby("app_name", as_index=False)
                 .agg(review_count=("review_text", "size"),
                      average_rating=("rating", "mean"),
                      average_word_count=("word_count", "mean"),
                      min_word_count=("word_count", "min"),
                      max_word_count=("word_count", "max"))
                 .sort_values("review_count", ascending=False))
    summary.to_csv(FINAL_DATASET_SUMMARY, index=False, encoding="utf-8")
    print(f"PROMOTED {len(data):,} reviews to {FINAL_RAW}")
    print(f"Updated existing report: {FINAL_DATASET_SUMMARY}")
    return True


def main():
    parser=argparse.ArgumentParser(description="Collect English-only reviews from Google Play and Apple; optionally promote the validated dataset.")
    parser.add_argument("--google-countries",default="gb,us,ca,au,in,ie",help="Comma-separated countries for Google Play")
    parser.add_argument("--apple-countries",default="gb,us,ca,au,ie,in,nz,sg,za,de,nl,ro",help="Comma-separated Apple storefronts")
    parser.add_argument("--google-per-country",type=int,default=4000)
    parser.add_argument("--apple-pages",type=int,default=10)
    parser.add_argument("--delay",type=float,default=1.2)
    parser.add_argument("--english-confidence",type=float,default=0.92)
    parser.add_argument("--min-words",type=int,default=5)
    parser.add_argument("--skip-google",action="store_true")
    parser.add_argument("--skip-apple",action="store_true")
    parser.add_argument("--promote",action="store_true",help="After validation, replace the existing raw dataset and update the existing app-level summary")
    args=parser.parse_args()
    if args.google_per_country<1 or not 1<=args.apple_pages<=10 or args.delay<0.5:
        parser.error("google-per-country >= 1, apple-pages 1..10, delay >= 0.5 are required")
    if not 0.5<=args.english_confidence<=1.0:
        parser.error("english-confidence must be between 0.5 and 1.0")
    if not CONFIG.exists():
        parser.error(f"Missing config: {CONFIG}")
    config=pd.read_csv(CONFIG, dtype=str, keep_default_na=False).to_dict("records")
    google_countries=[x.strip().lower() for x in args.google_countries.split(",") if x.strip()]
    apple_countries=[x.strip().lower() for x in args.apple_countries.split(",") if x.strip()]
    attempts=[]
    session=requests.Session()
    session.headers.update({"User-Agent":"Mozilla/5.0 (compatible; research-review-collection/1.0)","Accept":"application/json"})
    print(f"Configured apps: {len(config)}; Google storefronts: {google_countries}; Apple storefronts: {apple_countries}")
    print("Storefront results are cached across reruns. The existing final raw dataset is preserved unless --promote is given.")
    restricted=False
    try:
        for a in config:
            print(f"\nAPP: {a['app_name']}")
            if not args.skip_google and str(a.get('google_app_id','')).strip():
                for country in dict.fromkeys([str(a['preferred_country']).lower()]+google_countries):
                    status=collect_google(a,country,args.google_per_country,args.delay,200)
                    attempts.append(status)
            if not args.skip_apple and str(a.get('apple_app_id','')).strip():
                for country in dict.fromkeys([str(a['preferred_country']).lower()]+apple_countries):
                    try:
                        status=collect_apple(a,country,args.apple_pages,args.delay,session)
                        attempts.append(status)
                    except AccessStopped as exc:
                        print(f"Access restriction: {exc}. Stopping further requests.")
                        restricted=True
                        break
            if restricted:
                break
    except KeyboardInterrupt:
        print("Interrupted safely. Cached storefront files remain. Rerun the same command to resume.")
    finally:
        report = finalize(args.english_confidence,args.min_words,attempts)
    if args.promote and not restricted:
        promote_candidate(report)
    if restricted:
        print("Some storefronts were not completed because access was restricted.")

if __name__=="__main__":
    main()
