#!/usr/bin/env python3
from __future__ import annotations

import argparse, calendar, csv, gzip, hashlib, json, math, os, re, shutil, subprocess, sys, time, unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

csv.field_size_limit(sys.maxsize)
ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"
BACKUP_DIR = ROOT / "_backups"
FINAL_PATH = PROCESSED_DIR / "transactions.csv"
WORKLOG_PATH = ROOT / "transactions_db_export_worklog.md"
MANIFEST_PATH = OUTPUTS_DIR / "transactions_export_manifest.json"
QUALITY_REPORT_PATH = OUTPUTS_DIR / "transactions_quality_report.md"
ISSUES_PATH = OUTPUTS_DIR / "transactions_issues.csv"

FINAL_COLUMNS = ["transaction_id","complex_id","raw_complex_name","normalized_complex_name","complex_id_method","legal_dong_code","sgg_code","area_m2","floor","build_year","age_years","deal_date","deal_ym","trade_type","is_cancelled","reported_at","price_total","price_per_m2","target","complex_prev_price_per_m2","complex_prev_missing","prev_deal_gap_days"]
REQUIRED_FINAL = ["transaction_id","complex_id","raw_complex_name","normalized_complex_name","complex_id_method","legal_dong_code","sgg_code","area_m2","floor","build_year","age_years","deal_date","deal_ym","trade_type","is_cancelled","price_total","price_per_m2","target"]
ISSUE_COLUMNS = ["trade_id","transaction_id","severity","reason","deal_date","complex_id","detail"]
BASE_COLUMNS = ["trade_id","db_complex_id","deal_date","deal_amount","floor","excl_area","trade_source","trade_source_key","trade_apt_seq","trade_complex_pk","raw_ingest_id","deleted_at","complex_apt_seq","complex_complex_pk","complex_name","complex_trade_name","complex_use_date","parcel_pnu","evidence_sgg_cd","evidence_umd_cd","evidence_apt_seq","evidence_apt_name","evidence_match_path","raw_payload"]
LEGAL_DONG_RE = re.compile(r"^[0-9]{10}$")


def ensure_dirs():
    for p in [INTERIM_DIR, PROCESSED_DIR, OUTPUTS_DIR, BACKUP_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def run_id() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def db_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PGHOST", env.get("HOME_SEARCH_DB_HOST", "localhost"))
    env.setdefault("PGPORT", env.get("HOME_SEARCH_DB_PORT", "15432"))
    env.setdefault("PGDATABASE", env.get("HOME_SEARCH_DB_NAME", "home_search"))
    env.setdefault("PGUSER", env.get("HOME_SEARCH_DB_USERNAME", "home_search"))
    env.setdefault("PGPASSWORD", env.get("HOME_SEARCH_DB_PASSWORD", "home_search_local_password"))
    opts = ["-c", "default_transaction_read_only=on", "-c", f"statement_timeout={env.get('HOME_TRANSACTIONS_STATEMENT_TIMEOUT_MS','120000')}", "-c", f"lock_timeout={env.get('HOME_TRANSACTIONS_LOCK_TIMEOUT_MS','2000')}"]
    env["PGOPTIONS"] = (env.get("PGOPTIONS", "") + " " + " ".join(opts)).strip()
    return env


def psql_cmd() -> List[str]:
    return ["psql", "-X", "-q", "-v", "ON_ERROR_STOP=1"]


def psql_scalar(sql: str, timeout: int = 180) -> str:
    p = subprocess.run(psql_cmd() + ["-At", "-c", sql], env=db_env(), text=True, capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout).strip())
    return p.stdout.strip()


def month_ranges(start: date, end: date):
    cur = date(start.year, start.month, 1)
    while cur < end:
        last = calendar.monthrange(cur.year, cur.month)[1]
        nxt = date(cur.year, cur.month, last) + timedelta(days=1)
        yield max(cur, start), min(nxt, end)
        cur = nxt


def week_ranges(start: date, end: date):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=7), end)
        yield cur, nxt
        cur = nxt


def lit(d: date) -> str:
    return "'" + d.isoformat() + "'"


def export_sql(start: date, end: date) -> str:
    return f"""
COPY (
  SELECT t.id AS trade_id, t.complex_id AS db_complex_id, t.deal_date, t.deal_amount, t.floor, t.excl_area,
         t.source AS trade_source, t.source_key AS trade_source_key, t.apt_seq AS trade_apt_seq,
         t.complex_pk AS trade_complex_pk, t.raw_ingest_id, t.deleted_at,
         c.apt_seq AS complex_apt_seq, c.complex_pk AS complex_complex_pk, c.name AS complex_name,
         c.trade_name AS complex_trade_name, c.use_date AS complex_use_date, p.pnu AS parcel_pnu,
         e.sgg_cd AS evidence_sgg_cd, e.umd_cd AS evidence_umd_cd, e.apt_seq AS evidence_apt_seq,
         e.apt_name AS evidence_apt_name, e.match_path AS evidence_match_path, r.payload AS raw_payload
  FROM trade t
  LEFT JOIN complex c ON c.id = t.complex_id
  LEFT JOIN parcel p ON p.id = c.parcel_id
  LEFT JOIN raw_trade_ingest r ON r.id = t.raw_ingest_id
  LEFT JOIN LATERAL (
    SELECT sgg_cd, umd_cd, apt_seq, apt_name, match_path
    FROM trade_match_evidence e
    WHERE e.raw_ingest_id = t.raw_ingest_id
    ORDER BY e.id DESC
    LIMIT 1
  ) e ON true
  WHERE t.deal_date >= DATE {lit(start)} AND t.deal_date < DATE {lit(end)}
  ORDER BY t.deal_date, t.id
) TO STDOUT WITH CSV HEADER
""".strip()


def export_interval(start: date, end: date, out: Path) -> Dict[str, Any]:
    tmp = out.with_suffix(out.suffix + ".tmp")
    t0 = time.monotonic()
    p = subprocess.Popen(psql_cmd() + ["-c", export_sql(start, end)], env=db_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    rows, first = 0, True
    try:
        with gzip.open(tmp, "wb") as f:
            assert p.stdout is not None
            for line in p.stdout:
                f.write(line)
                if first:
                    first = False
                else:
                    rows += 1
        err = p.stderr.read().decode("utf-8", "replace") if p.stderr else ""
        rc = p.wait()
    except Exception:
        p.kill(); tmp.unlink(missing_ok=True); raise
    if rc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(err.strip() or f"psql exit {rc}")
    tmp.replace(out)
    return {"start": start.isoformat(), "end_exclusive": end.isoformat(), "rows": rows, "seconds": round(time.monotonic()-t0,3), "path": str(out)}


def export_chunks() -> Tuple[List[Path], List[Dict[str, Any]]]:
    paths, logs = [], []
    end = date.today() + timedelta(days=1)
    for ms, me in month_ranges(date(2017,1,1), end):
        mp = INTERIM_DIR / f"transactions_base_{ms.year:04d}_{ms.month:02d}.csv.gz"
        try:
            info = export_interval(ms, me, mp); info["granularity"] = "month"
            logs.append(info); paths.append(mp); continue
        except Exception as e:
            logs.append({"start": ms.isoformat(), "end_exclusive": me.isoformat(), "granularity": "month", "error": str(e)})
        for ws, we in week_ranges(ms, me):
            wp = INTERIM_DIR / f"transactions_base_{ws.isoformat()}_{(we-timedelta(days=1)).isoformat()}.csv.gz"
            last = None
            for attempt in range(1,4):
                try:
                    info = export_interval(ws, we, wp); info.update({"granularity":"week", "attempt":attempt})
                    logs.append(info); paths.append(wp); last = None; break
                except Exception as e:
                    last = e; logs.append({"start": ws.isoformat(), "end_exclusive": we.isoformat(), "granularity": "week", "attempt": attempt, "error": str(e)}); time.sleep(1)
            if last:
                raise RuntimeError(f"chunk failed after 3 attempts: {ws}..{we}: {last}")
    return paths, logs


def clean(v: Any) -> str:
    if v is None: return ""
    s = str(v).strip()
    return "" if s.lower() in {"none","null","nan"} else s


def norm_name(v: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", clean(v))).strip()


def parse_date(v: Any) -> Optional[date]:
    s = clean(v)
    if not s: return None
    for fmt in ("%Y-%m-%d","%Y%m%d","%Y.%m.%d","%y.%m.%d"):
        try: return datetime.strptime(s, fmt).date()
        except ValueError: pass
    return None


def dec(v: Any) -> Optional[Decimal]:
    s = clean(v).replace(",", "")
    if not s: return None
    try: return Decimal(s)
    except InvalidOperation: return None


def parse_int(v: Any) -> Optional[int]:
    d = dec(v)
    return int(d) if d is not None else None


def raw_json(payload: str) -> Dict[str, Any]:
    try:
        x = json.loads(clean(payload)); return x if isinstance(x, dict) else {}
    except Exception:
        return {}


def rget(raw: Dict[str, Any], k: str) -> str:
    return clean(raw.get(k))


def legal_code(row: Dict[str,str], raw: Dict[str,Any]) -> str:
    sgg = clean(row.get("evidence_sgg_cd")) or rget(raw,"sggCd")
    umd = clean(row.get("evidence_umd_cd")) or rget(raw,"umdCd")
    if sgg and umd and sgg.isdigit() and umd.isdigit():
        return f"{int(sgg):05d}{int(umd):05d}"
    pnu = clean(row.get("parcel_pnu"))
    return pnu[:10] if len(pnu) >= 10 and pnu[:10].isdigit() else ""


def build_year(row: Dict[str,str], raw: Dict[str,Any]) -> Tuple[Optional[int], str]:
    use = parse_date(row.get("complex_use_date"))
    if use: return use.year, "use_date"
    y = parse_int(raw.get("buildYear"))
    return (y, "raw_buildYear") if y else (None, "missing")


def make_tid(deal_dt: date, complex_id: str, source_key: str, trade_id: str) -> str:
    digest = hashlib.sha256((source_key or trade_id).encode()).hexdigest()[:8]
    return f"{deal_dt.isoformat()}_{complex_id}_{digest}"


def prepare(row: Dict[str,str]) -> Dict[str,Any]:
    raw = raw_json(row.get("raw_payload", ""))
    deal_dt, area, total, floor = parse_date(row.get("deal_date")), dec(row.get("excl_area")), dec(row.get("deal_amount")), parse_int(row.get("floor"))
    apt_seq = clean(row.get("trade_apt_seq")) or clean(row.get("complex_apt_seq")) or clean(row.get("evidence_apt_seq")) or rget(raw,"aptSeq")
    cpk = clean(row.get("trade_complex_pk")) or clean(row.get("complex_complex_pk"))
    raw_name = rget(raw,"aptNm") or clean(row.get("evidence_apt_name")) or clean(row.get("complex_trade_name")) or clean(row.get("complex_name"))
    normalized = norm_name(raw_name)
    legal = legal_code(row, raw)
    if apt_seq: complex_id, method = apt_seq, "apt_code_join"
    elif cpk: complex_id, method = cpk, "complex_pk_join"
    elif legal and normalized: complex_id, method = f"{legal}_{normalized}", "legal_dong_name_fallback"
    else: complex_id, method = "", "legal_dong_name_fallback"
    by, by_src = build_year(row, raw)
    ppm = (total / area) if total is not None and area is not None and area > 0 else None
    target = math.log(float(ppm)) if ppm is not None and ppm > 0 else None
    age = (deal_dt.year - by) if deal_dt and by else None
    source_key = clean(row.get("trade_source_key")) or clean(row.get("raw_ingest_id"))
    tid = make_tid(deal_dt, complex_id, source_key, clean(row.get("trade_id"))) if deal_dt and complex_id else ""
    cancelled = 1 if clean(row.get("deleted_at")) or rget(raw,"cdealType") == "O" else 0
    reported = parse_date(raw.get("rgstDate"))
    return {"trade_id":clean(row.get("trade_id")),"transaction_id":tid,"complex_id":complex_id,"raw_complex_name":raw_name,"normalized_complex_name":normalized,"complex_id_method":method,"legal_dong_code":legal,"sgg_code":legal[:5] if len(legal)>=5 else "","area_m2":area,"floor":floor,"build_year":by,"build_year_source":by_src,"age_years":age,"deal_date_obj":deal_dt,"deal_date":deal_dt.isoformat() if deal_dt else "","deal_ym":deal_dt.isoformat()[:7] if deal_dt else "","trade_type":rget(raw,"dealingGbn") or "unknown","is_cancelled":cancelled,"reported_at":reported.isoformat() if reported else "","price_total":total,"price_per_m2":ppm,"target":target,"complex_prev_price_per_m2":None,"complex_prev_missing":1,"prev_deal_gap_days":None}


def split_name(d: date) -> str:
    if d <= date(2023,12,31): return "train"
    if d <= date(2024,12,31): return "valid"
    if d <= date(2025,12,31): return "test"
    return "recent_holdout"


def exclusion(r: Dict[str,Any]) -> Optional[str]:
    checks = [(not r["transaction_id"],"missing_transaction_id"),(not r["complex_id"],"missing_complex_id"),(not r["deal_date_obj"],"missing_deal_date"),(not LEGAL_DONG_RE.match(r["legal_dong_code"] or ""),"missing_or_invalid_legal_dong_code"),(not r["raw_complex_name"] or not r["normalized_complex_name"],"missing_complex_name"),(r["price_total"] is None or r["price_total"] <= 0,"invalid_price_total"),(r["area_m2"] is None or r["area_m2"] <= 0,"invalid_area_m2"),(r["floor"] is None,"missing_floor"),(r["build_year"] is None,"missing_build_year"),(r["age_years"] is None or r["age_years"] < 0,"invalid_age_years"),(r["target"] is None,"invalid_target")]
    for bad, reason in checks:
        if bad: return reason
    return None

@dataclass
class Hist:
    area: Decimal
    deal_date: date
    ppm: Decimal


def find_prev(hist: Dict[str,Dict[int,List[Hist]]], cid: str, area: Decimal, d: date) -> Tuple[Optional[Decimal], Optional[int]]:
    buckets = hist.get(cid)
    if not buckets: return None, None
    lo, hi = area * Decimal("0.9"), area * Decimal("1.1")
    best = None
    for b in range(math.floor(float(lo)), math.floor(float(hi)) + 1):
        items = buckets.get(b)
        if not items: continue
        for item in reversed(items):
            if item.deal_date >= d: continue
            if lo <= item.area <= hi:
                if best is None or item.deal_date > best.deal_date: best = item
                break
    return (best.ppm, (d - best.deal_date).days) if best else (None, None)


def add_hist(hist: Dict[str,Dict[int,List[Hist]]], r: Dict[str,Any]) -> None:
    if r["is_cancelled"] or not r["complex_id"] or not r["deal_date_obj"]: return
    if r["area_m2"] is None or r["area_m2"] <= 0 or r["price_per_m2"] is None or r["price_per_m2"] <= 0: return
    hist[r["complex_id"]][math.floor(float(r["area_m2"]))].append(Hist(r["area_m2"], r["deal_date_obj"], r["price_per_m2"]))


def fmt_dec(v: Optional[Decimal], n=6) -> str:
    return "" if v is None else f"{float(v):.{n}f}"


def out_row(r: Dict[str,Any]) -> Dict[str,str]:
    return {"transaction_id":r["transaction_id"],"complex_id":r["complex_id"],"raw_complex_name":r["raw_complex_name"],"normalized_complex_name":r["normalized_complex_name"],"complex_id_method":r["complex_id_method"],"legal_dong_code":r["legal_dong_code"],"sgg_code":r["sgg_code"],"area_m2":fmt_dec(r["area_m2"]),"floor":"" if r["floor"] is None else str(r["floor"]),"build_year":"" if r["build_year"] is None else str(r["build_year"]),"age_years":"" if r["age_years"] is None else str(r["age_years"]),"deal_date":r["deal_date"],"deal_ym":r["deal_ym"],"trade_type":r["trade_type"],"is_cancelled":str(r["is_cancelled"]),"reported_at":r["reported_at"],"price_total":fmt_dec(r["price_total"],0),"price_per_m2":fmt_dec(r["price_per_m2"],12),"target":"" if r["target"] is None else f"{r['target']:.12f}","complex_prev_price_per_m2":fmt_dec(r["complex_prev_price_per_m2"],12),"complex_prev_missing":str(r["complex_prev_missing"]),"prev_deal_gap_days":"" if r["prev_deal_gap_days"] is None else str(r["prev_deal_gap_days"])}


def iter_rows(paths: List[Path]):
    for p in sorted(paths):
        with gzip.open(p, "rt", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            missing = [c for c in BASE_COLUMNS if c not in (reader.fieldnames or [])]
            if missing: raise RuntimeError(f"base chunk missing columns {missing}: {p}")
            yield from reader


def build_from_chunks(paths: List[Path], tmp: Path, rid: str) -> Dict[str,Any]:
    hist: Dict[str,Dict[int,List[Hist]]] = defaultdict(lambda: defaultdict(list))
    seen, current, day = set(), None, []
    stats: Dict[str,Any] = {"base_rows":0,"final_rows":0,"excluded_rows":0,"warn_rows":0,"cancelled_rows":0,"reported_at_null_rows":0,"prev_missing_rows":0,"build_year_source":defaultdict(int),"trade_type":defaultdict(int),"split_counts":defaultdict(int),"as_of_checked_rows":0,"as_of_failures":0,"run_id":rid}
    with tmp.open("w", encoding="utf-8", newline="") as outf, ISSUES_PATH.open("w", encoding="utf-8", newline="") as issuef:
        writer = csv.DictWriter(outf, fieldnames=FINAL_COLUMNS); writer.writeheader()
        iw = csv.DictWriter(issuef, fieldnames=ISSUE_COLUMNS); iw.writeheader()
        def issue(r, sev, reason, detail=""):
            stats["excluded_rows" if sev == "exclude" else "warn_rows"] += 1
            iw.writerow({"trade_id":r.get("trade_id",""),"transaction_id":r.get("transaction_id",""),"severity":sev,"reason":reason,"deal_date":r.get("deal_date",""),"complex_id":r.get("complex_id",""),"detail":detail})
        def flush(records):
            if not records: return
            for r in records:
                if r["area_m2"] is not None and r["deal_date_obj"] and r["complex_id"]:
                    pp, gap = find_prev(hist, r["complex_id"], r["area_m2"], r["deal_date_obj"]); stats["as_of_checked_rows"] += 1
                    if pp is None: stats["prev_missing_rows"] += 1
                    else: r["complex_prev_price_per_m2"], r["complex_prev_missing"], r["prev_deal_gap_days"] = pp, 0, gap
                if not r["deal_date_obj"] or r["deal_date_obj"] < date(2019,1,1): continue
                reason = exclusion(r)
                if reason: issue(r, "exclude", reason); continue
                if r["floor"] is not None and r["floor"] < 0: issue(r, "warn", "negative_floor", "retained")
                tid = r["transaction_id"]
                if tid in seen: tid = f"{tid}_{r['trade_id']}"; r["transaction_id"] = tid
                if tid in seen: raise RuntimeError(f"duplicate transaction_id after collision handling: {tid}")
                seen.add(tid); writer.writerow(out_row(r)); stats["final_rows"] += 1
                if r["is_cancelled"]: stats["cancelled_rows"] += 1
                if not r["reported_at"]: stats["reported_at_null_rows"] += 1
                stats["build_year_source"][r["build_year_source"]] += 1; stats["trade_type"][r["trade_type"]] += 1; stats["split_counts"][split_name(r["deal_date_obj"])] += 1
            for r in records: add_hist(hist, r)
        for row in iter_rows(paths):
            stats["base_rows"] += 1; r = prepare(row); d = r["deal_date_obj"]
            if current is None: current = d
            if d != current: flush(day); day = []; current = d
            day.append(r)
        flush(day)
    for k in ["build_year_source","trade_type","split_counts"]: stats[k] = dict(stats[k])
    return stats


def validate_csv(path: Path) -> Dict[str,Any]:
    res: Dict[str,Any] = {"path":str(path),"rows":0,"schema_ok":False,"duplicate_ids":0,"required_nulls":defaultdict(int),"value_errors":defaultdict(int),"split_counts":defaultdict(int),"reported_at_null_rows":0}
    seen = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f); res["columns"] = reader.fieldnames or []; res["schema_ok"] = res["columns"] == FINAL_COLUMNS
        for row in reader:
            res["rows"] += 1; tid = row.get("transaction_id", "")
            if tid in seen: res["duplicate_ids"] += 1
            seen.add(tid)
            for c in REQUIRED_FINAL:
                if clean(row.get(c)) == "": res["required_nulls"][c] += 1
            if clean(row.get("reported_at")) == "": res["reported_at_null_rows"] += 1
            legal = row.get("legal_dong_code", "")
            if not LEGAL_DONG_RE.match(legal): res["value_errors"]["legal_dong_code"] += 1
            if row.get("sgg_code") != legal[:5]: res["value_errors"]["sgg_code"] += 1
            if row.get("deal_ym") != row.get("deal_date", "")[:7]: res["value_errors"]["deal_ym"] += 1
            try:
                area,total,ppm,target = Decimal(row["area_m2"]), Decimal(row["price_total"]), Decimal(row["price_per_m2"]), float(row["target"])
                if area <= 0 or total <= 0 or ppm <= 0: res["value_errors"]["positive_price_area"] += 1
                if abs(float((total / area) - ppm)) > 1e-6: res["value_errors"]["price_per_m2"] += 1
                if abs(math.log(float(ppm)) - target) > 1e-9: res["value_errors"]["target"] += 1
            except Exception: res["value_errors"]["numeric_parse"] += 1
            try:
                if int(row["age_years"]) < 0: res["value_errors"]["age_years"] += 1
            except Exception: res["value_errors"]["age_years_parse"] += 1
            miss, prev, gap = row.get("complex_prev_missing"), clean(row.get("complex_prev_price_per_m2")), clean(row.get("prev_deal_gap_days"))
            if miss == "1" and (prev or gap): res["value_errors"]["prev_missing_consistency"] += 1
            if miss == "0" and (not prev or not gap): res["value_errors"]["prev_present_consistency"] += 1
            if miss not in {"0","1"}: res["value_errors"]["prev_missing_flag"] += 1
            d = parse_date(row.get("deal_date"))
            if d: res["split_counts"][split_name(d)] += 1
    for k in ["required_nulls","value_errors","split_counts"]: res[k] = dict(res[k])
    fail = (not res["schema_ok"] or res["duplicate_ids"] > 0 or any(res["required_nulls"].values()) or any(res["value_errors"].values()))
    res["grade"] = "Fail" if fail else ("Partial" if res["reported_at_null_rows"] else "Pass")
    return res


def db_summary() -> Dict[str,Any]:
    db,user,host,port = psql_scalar("SELECT current_database() || '|' || current_user || '|' || inet_server_addr() || '|' || inet_server_port()").split("|")
    ro = psql_scalar("SHOW default_transaction_read_only")
    cov = psql_scalar("""
SELECT min(deal_date)::text || '|' || max(deal_date)::text || '|' || count(*) FILTER (WHERE deal_amount IS NULL)::text || '|' || count(*) FILTER (WHERE excl_area IS NULL)::text || '|' || count(*) FILTER (WHERE apt_seq IS NULL)::text || '|' || count(*) FILTER (WHERE complex_id IS NULL)::text
FROM trade WHERE deal_date >= DATE '2017-01-01'
""").split("|")
    return {"database":db,"user":user,"host":host,"port":port,"read_only":ro,"coverage_2017_plus":{"min_deal_date":cov[0],"max_deal_date":cov[1],"deal_amount_nulls":int(cov[2]),"excl_area_nulls":int(cov[3]),"apt_seq_nulls":int(cov[4]),"complex_id_nulls":int(cov[5])}}


def write_manifest(m: Dict[str,Any]):
    MANIFEST_PATH.write_text(json.dumps(m, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_worklog(m: Dict[str,Any], v: Optional[Dict[str,Any]]=None):
    s = m.get("stats", {})
    lines = ["# transactions.csv DB export worklog","",f"- 실행 시작: {m.get('started_at')}",f"- 실행 종료: {m.get('finished_at')}",f"- run_id: `{m.get('run_id')}`",f"- DB: `{m.get('db',{}).get('database')}@{m.get('db',{}).get('host')}:{m.get('db',{}).get('port')}`",f"- read-only: `{m.get('db',{}).get('read_only')}`","","## Chunk export"]
    for c in m.get("chunks", []):
        lines.append(f"- {c.get('start')}..{c.get('end_exclusive')} `{c.get('granularity')}` " + (f"실패: {c.get('error')}" if 'error' in c else f"rows={c.get('rows')} seconds={c.get('seconds')}"))
    lines += ["","## Quality signals",f"- base rows: {s.get('base_rows')}",f"- final rows: {s.get('final_rows')}",f"- build_year source: {s.get('build_year_source')}",f"- reported_at null rows: {s.get('reported_at_null_rows')}",f"- trade_type distribution: {s.get('trade_type')}",f"- cancelled rows: {s.get('cancelled_rows')}",f"- excluded rows: {s.get('excluded_rows')}",f"- warn rows: {s.get('warn_rows')}",f"- final path: `{m.get('final_path')}`",f"- final size bytes: {m.get('final_size_bytes')}","","## Next retry point","`/goal Home Search DB 기반 transactions.csv를 완성한다. 기존 manifest와 failed artifact를 읽고 마지막 성공 chunk부터 이어서 실행하며, quality report의 실패 사유를 해결해 최종 transactions.csv 검증을 통과시킨다.`"]
    if v: lines += ["","## Validation",f"- grade: {v.get('grade')}",f"- rows: {v.get('rows')}"]
    WORKLOG_PATH.write_text("\n".join(lines)+"\n", encoding="utf-8")


def write_report(v: Dict[str,Any], m: Optional[Dict[str,Any]]=None):
    st = (m or {}).get("stats", {})
    checks = [("컬럼 순서/개수", v.get("schema_ok")), ("transaction_id unique", v.get("duplicate_ids") == 0), ("필수 feature/label null 없음", not any(v.get("required_nulls",{}).values())), ("price_per_m2 계산 일치", v.get("value_errors",{}).get("price_per_m2",0)==0), ("target 계산 일치", v.get("value_errors",{}).get("target",0)==0), ("deal_ym 일치", v.get("value_errors",{}).get("deal_ym",0)==0), ("sgg_code 일치", v.get("value_errors",{}).get("sgg_code",0)==0), ("legal_dong_code 10자리", v.get("value_errors",{}).get("legal_dong_code",0)==0), ("age_years >= 0", v.get("value_errors",{}).get("age_years",0)==0), ("complex_prev_missing consistency", v.get("value_errors",{}).get("prev_missing_consistency",0)==0 and v.get("value_errors",{}).get("prev_present_consistency",0)==0), ("as-of 누수 없음", st.get("as_of_failures",0)==0)]
    lines = ["# transactions.csv quality report","",f"- 품질 등급: `{v.get('grade')}`",f"- rows: {v.get('rows')}",f"- path: `{v.get('path')}`","","## 지적사항"]
    lines.append("- Fail 항목이 있습니다. 아래 검증 세부 정보를 확인하세요." if v.get("grade") == "Fail" else ("- optional coverage 이슈가 있습니다. `reported_at` null은 v1 허용 범위입니다." if v.get("grade") == "Partial" else "- none"))
    lines += ["","## 검증 근거 확인"] + [f"- {label}: {'pass' if ok else 'fail'}" for label, ok in checks]
    lines += ["","## 검증 공백","- standalone `--validate`는 CSV에 prev source date가 없으므로 as-of 검증을 재계산하지 않습니다. `--build` 중 동일 일자 buffer 처리와 build-time assertion 결과를 manifest에 기록합니다.","","## Split distribution"]
    for k in ["train","valid","test","recent_holdout"]: lines.append(f"- {k}: {v.get('split_counts',{}).get(k,0)}")
    lines += ["","## Detail",f"- duplicate_ids: {v.get('duplicate_ids')}",f"- required_nulls: {v.get('required_nulls')}",f"- value_errors: {v.get('value_errors')}",f"- reported_at_null_rows: {v.get('reported_at_null_rows')}",f"- issues_csv: `{ISSUES_PATH}`",f"- manifest: `{MANIFEST_PATH}`"]
    QUALITY_REPORT_PATH.write_text("\n".join(lines)+"\n", encoding="utf-8")


def dry_run():
    ensure_dirs(); m = {"mode":"dry-run","run_id":run_id(),"started_at":now_utc()}; m["db"] = db_summary(); m["finished_at"] = now_utc(); write_manifest(m); write_worklog(m); print(json.dumps(m, ensure_ascii=False, indent=2))


def build():
    ensure_dirs(); rid = run_id(); tmp = PROCESSED_DIR / f"transactions.csv.tmp.{rid}"; failed = PROCESSED_DIR / f"transactions.csv.failed-{rid}"; m: Dict[str,Any] = {"mode":"build","run_id":rid,"started_at":now_utc()}
    try:
        m["db"] = db_summary(); paths, logs = export_chunks(); m["chunks"] = logs; m["stats"] = build_from_chunks(paths, tmp, rid); v = validate_csv(tmp); m["validation"] = v
        if v["grade"] == "Fail": tmp.replace(failed); m["failed_path"] = str(failed); raise RuntimeError(f"validation failed; tmp moved to {failed}")
        if FINAL_PATH.exists(): backup = BACKUP_DIR / f"transactions_{rid}.csv"; shutil.copy2(FINAL_PATH, backup); m["backup_path"] = str(backup)
        os.replace(tmp, FINAL_PATH); m["final_path"] = str(FINAL_PATH); m["final_size_bytes"] = FINAL_PATH.stat().st_size
        v = validate_csv(FINAL_PATH); m["validation"] = v; m["finished_at"] = now_utc(); write_manifest(m); write_report(v,m); write_worklog(m,v); print(f"built {FINAL_PATH} rows={v['rows']} grade={v['grade']}")
    except Exception as e:
        if tmp.exists(): tmp.replace(failed); m["failed_path"] = str(failed)
        m["error"] = str(e); m["finished_at"] = now_utc(); write_manifest(m); write_worklog(m, m.get("validation")); raise


def validate(path: Path):
    ensure_dirs(); v = validate_csv(path); m = {}
    if MANIFEST_PATH.exists():
        try: m = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception: m = {}
    write_report(v,m); print(json.dumps(v, ensure_ascii=False, indent=2))
    if v["grade"] == "Fail": raise SystemExit(1)


def self_test():
    def rec(tid, cid, d, area, total, cancelled=0, build=2000):
        a, p = Decimal(area), Decimal(total); ppm = p / a
        return {"trade_id":tid,"transaction_id":f"tx-{tid}","complex_id":cid,"raw_complex_name":"Apt","normalized_complex_name":"Apt","complex_id_method":"apt_code_join","legal_dong_code":"1168010100","sgg_code":"11680","area_m2":a,"floor":1,"build_year":build,"build_year_source":"use_date","age_years":d.year-build,"deal_date_obj":d,"deal_date":d.isoformat(),"deal_ym":d.isoformat()[:7],"trade_type":"unknown","is_cancelled":cancelled,"reported_at":"","price_total":p,"price_per_m2":ppm,"target":math.log(float(ppm)),"complex_prev_price_per_m2":None,"complex_prev_missing":1,"prev_deal_gap_days":None}
    hist: Dict[str,Dict[int,List[Hist]]] = defaultdict(lambda: defaultdict(list))
    first, same = rec("1","C1",date(2020,1,1),"84","84000"), rec("2","C1",date(2020,1,1),"84","85000")
    assert find_prev(hist,"C1",first["area_m2"],first["deal_date_obj"])[0] is None
    add_hist(hist, first); add_hist(hist, same)
    prev, gap = find_prev(hist,"C1",Decimal("84"),date(2020,1,2)); assert prev == same["price_per_m2"] and gap == 1
    assert find_prev(hist,"C1",Decimal("120"),date(2020,1,3))[0] is None
    c = rec("3","C2",date(2020,1,1),"84","84000",cancelled=1); add_hist(hist,c); assert find_prev(hist,"C2",Decimal("84"),date(2020,1,2))[0] is None
    assert build_year({"complex_use_date":""},{"buildYear":1999}) == (1999,"raw_buildYear")
    assert parse_date("26.06.12") == date(2026, 6, 12)
    assert exclusion(rec("4","C3",date(2020,1,1),"84","84000",build=2021)) == "invalid_age_years"
    print("self-test pass")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--build", action="store_true"); ap.add_argument("--validate", type=Path); ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args();
    if sum(bool(x) for x in [a.dry_run,a.build,a.validate,a.self_test]) != 1: ap.error("choose exactly one mode")
    if a.self_test: self_test()
    elif a.dry_run: dry_run()
    elif a.build: build()
    else: validate(a.validate)

if __name__ == "__main__": main()
