#!/usr/bin/env python3
"""
Hugging Face에서 documents.csv를 다운로드하고,
각 행의 url에서 PDF 파일을 다운로드하여 data/raw_pdfs/에 저장하며,
documents.csv에 'downloaded' 열(True/False)을 기록 및 갱신하고,
최종적으로 다운로드에 실패한 row(file_name, url)만 모아 data/failed.csv로 저장하는 스크립트.

동작 흐름:
1. documents.csv 준비: 없을 경우 Hugging Face에서 자동 다운로드.
2. [사전 대조]: data/raw_pdfs 디렉토리의 실제 파일들과 documents.csv의 file_name을 대조.
   - 실제 존재하는 파일은 documents.csv의 downloaded 열에 무조건 True로 반영/갱신.
   - 존재하지 않는 파일은 False로 반영.
   - 대조 결과를 documents.csv 및 failed.csv에 즉시 저장.
3. [선별 다운로드/재시도]:
   - downloaded=True인 항목은 건너뜀 (SKIP).
   - downloaded=False인 항목만 선별하여 다운로드 재시도 (RETRY).
   - 다운로드 성공 시 downloaded=True로 갱신하여 저장.
4. [최종 failed.csv 갱신]:
   - 최종적으로 실패(downloaded!=True)한 항목들만 모아 file_name, url 컬럼으로 failed.csv 생성.
"""

import argparse
import csv
import http.cookiejar
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# 기본 디렉토리 설정 (스크립트 위치 기준: ../data, ../data/raw_pdfs, ../data/failed.csv)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CSV_PATH = DEFAULT_DATA_DIR / "documents.csv"
DEFAULT_FAILED_CSV_PATH = DEFAULT_DATA_DIR / "failed.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "raw_pdfs"

HF_DATASET_REPO = "allganize/RAG-Evaluation-Dataset-KO"
HF_CSV_FILENAME = "documents.csv"
HF_DIRECT_URL = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main/{HF_CSV_FILENAME}"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}


def nfc(text: str) -> str:
    """macOS NFD(자모 분리) 및 NFC 유니코드 문자열 정규화."""
    return unicodedata.normalize("NFC", text.strip()) if text else ""


def create_ssl_context(insecure: bool = False) -> ssl.SSLContext:
    if insecure:
        return ssl._create_unverified_context()
    return ssl.create_default_context()


def get_opener(insecure: bool = False):
    """쿠키와 SSL 설정을 지원하는 urllib opener 생성."""
    cookie_jar = http.cookiejar.CookieJar()
    cookie_handler = urllib.request.HTTPCookieProcessor(cookie_jar)
    ctx = create_ssl_context(insecure=insecure)
    https_handler = urllib.request.HTTPSHandler(context=ctx)
    opener = urllib.request.build_opener(cookie_handler, https_handler)
    return opener


def download_documents_csv(data_dir: Path, csv_path: Path, force: bool = False):
    """Hugging Face에서 documents.csv를 다운로드합니다."""
    data_dir.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() and not force:
        print(f"[1단계] documents.csv 확인 완료: {csv_path}")
        return

    print(f"[1단계] Hugging Face에서 documents.csv 다운로드 시작...")

    # 방법 1: hf CLI 사용
    hf_cmd = shutil.which("hf")
    if hf_cmd:
        cmd = [
            hf_cmd,
            "download",
            f"hf://datasets/{HF_DATASET_REPO}/{HF_CSV_FILENAME}",
            "--local-dir",
            str(data_dir),
        ]
        print(f"       명령어 실행: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            if csv_path.exists():
                print(f"       hf CLI 다운로드 완료: {csv_path}")
                return
        except Exception as e:
            print(f"       hf CLI 실행 실패 ({e}), 다른 방법 시도 중...")

    # 방법 2: huggingface_hub 라이브러리 사용
    try:
        from huggingface_hub import hf_hub_download
        print(f"       huggingface_hub 라이브러리 사용하여 다운로드 시도...")
        downloaded = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=HF_CSV_FILENAME,
            repo_type="dataset",
            local_dir=str(data_dir),
        )
        if Path(downloaded).exists():
            print(f"       huggingface_hub 다운로드 완료: {downloaded}")
            return
    except ImportError:
        pass
    except Exception as e:
        print(f"       huggingface_hub 다운로드 실패 ({e}), 직접 HTTP 다운로드 시도...")

    # 방법 3: urllib 직접 다운로드
    print(f"       직접 HTTP 다운로드 시도: {HF_DIRECT_URL}")
    req = urllib.request.Request(HF_DIRECT_URL, headers=DEFAULT_HEADERS)
    ctx = create_ssl_context()
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        content = resp.read()
        with open(csv_path, "wb") as f:
            f.write(content)
    print(f"       HTTP 다운로드 완료: {csv_path} ({len(content):,} bytes)")


def fetch_url(url: str, timeout: int = 30, insecure: bool = False, referer: str = None):
    """지정된 URL에서 바이너리 데이터를 다운로드합니다."""
    headers = dict(DEFAULT_HEADERS)
    if referer:
        headers["Referer"] = referer

    req = urllib.request.Request(url, headers=headers)
    opener = get_opener(insecure=insecure)
    try:
        with opener.open(req, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            data = response.read()
            final_url = response.geturl()
            return data, content_type, final_url
    except (ssl.SSLError, urllib.error.URLError) as e:
        if not insecure and ("certificate" in str(e).lower() or "ssl" in str(e).lower()):
            opener_insecure = get_opener(insecure=True)
            with opener_insecure.open(req, timeout=timeout) as response:
                content_type = response.headers.get("Content-Type", "").lower()
                data = response.read()
                final_url = response.geturl()
                return data, content_type, final_url
        raise e


def try_find_pdf_link_in_html(html_text: str, base_url: str):
    """HTML 페이지에서 PDF 첨부파일 다운로드 링크를 탐색합니다."""
    # 대법원(scourt) 등 직접 파일 도메인 링크 우선 탐색
    scourt_match = re.search(r'https?://file\.scourt\.go\.kr/[^\s"\'<>]+\.pdf', html_text)
    if scourt_match:
        return scourt_match.group(0)

    # 일반적인 첨부파일 다운로드 링크 패턴
    patterns = [
        r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']',
        r'href=["\']([^"\']*(?:download|fileDown|downDomainFile|file_down|attachment|attach)[^"\']*)["\']',
        r'https?://[^\s"\'<>]+\.pdf(?:\?[^\s"\'<>]*)?',
    ]

    for pat in patterns:
        for match in re.finditer(pat, html_text, re.IGNORECASE):
            link = match.group(1 if match.groups() else 0).replace("&amp;", "&")
            if link.startswith("javascript:") or link.startswith("#"):
                continue
            return urllib.parse.urljoin(base_url, link)
    return None


def save_failed_csv(rows: list, failed_csv_path: Path):
    """실패한(미다운로드) row만 선별하여 file_name, url로 구성된 failed.csv를 저장합니다."""
    failed_rows = [
        {"file_name": r.get("file_name", "").strip(), "url": r.get("url", "").strip()}
        for r in rows
        if str(r.get("downloaded", "")).strip().lower() != "true"
    ]
    failed_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(failed_csv_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "url"])
        writer.writeheader()
        writer.writerows(failed_rows)
    print(f"[실패 목록 파일] 총 {len(failed_rows)}개의 실패 항목이 {failed_csv_path}에 저장되었습니다.")


# 공식 벤치마크 매핑 테이블(pdf_mapping.json) 로드
KNOWN_MAPPING = {}
mapping_file = PROJECT_ROOT / "ref" / "RAG-Evaluation" / "data" / "pdf_mapping.json"
if not mapping_file.exists():
    mapping_file = PROJECT_ROOT / "data" / "pdf_mapping.json"
if mapping_file.exists():
    try:
        with open(mapping_file, "r", encoding="utf-8") as f:
            for k, v in json.load(f).items():
                KNOWN_MAPPING[nfc(k).lower()] = nfc(v).lower()
                KNOWN_MAPPING[nfc(v).lower()] = nfc(k).lower()
    except Exception:
        pass


def is_filename_match(csv_name: str, disk_name: str) -> bool:
    """
    documents.csv의 file_name과 디스크 상의 파일명이 유사하거나 동일한지 검사.
    - 예: '130292099630937500_KIFVIP2013-10.pdf' <---> 'KIFVIP2013-10.pdf' 인정
    """
    c_raw = nfc(csv_name)
    d_raw = nfc(disk_name)

    c_low = c_raw.lower()
    d_low = d_raw.lower()

    # 1. 완전 일치 (대소문자/정규화 무시)
    if c_low == d_low:
        return True

    # 2. 공식 벤치마크 매핑 테이블 일치
    if KNOWN_MAPPING.get(c_low) == d_low or KNOWN_MAPPING.get(d_low) == c_low:
        return True

    c_stem = re.sub(r"\.pdf$", "", c_raw, flags=re.IGNORECASE)
    d_stem = re.sub(r"\.pdf$", "", d_raw, flags=re.IGNORECASE)

    # 3. 앞부분 숫자/타임스탬프 접두사(예: 130292099630937500_) 제거 후 비교
    c_clean = re.sub(r"^[0-9a-fA-F]{8,}_", "", c_stem)
    d_clean = re.sub(r"^[0-9a-fA-F]{8,}_", "", d_stem)

    if (
        c_clean.lower() == d_clean.lower()
        or c_clean.lower() == d_stem.lower()
        or c_stem.lower() == d_clean.lower()
    ):
        return True

    # [별첨]과 본문 문서가 혼동되지 않도록 방어
    if ("별첨" in c_stem) != ("별첨" in d_stem):
        return False

    # 4. 공백 및 특수문자 제거 후 정규화 비교
    c_norm = re.sub(r"[\s_\-+()\[\]★ㆍᆞ\.,~]", "", c_clean).lower()
    d_norm = re.sub(r"[\s_\-+()\[\]★ㆍᆞ\.,~]", "", d_clean).lower()

    if c_norm == d_norm:
        return True

    # 5. 한쪽이 다른 쪽의 핵심 단어/문자열을 포함하는 경우 (최소 6글자 이상)
    if len(c_norm) >= 6 and len(d_norm) >= 6:
        if c_norm in d_norm or d_norm in c_norm:
            # 문서 번호(예: _2 vs _3) 충돌 방지: 숫자가 포함되어 있으면 숫자 구성 일치 확인
            c_nums = re.findall(r"\d+", c_clean)
            d_nums = re.findall(r"\d+", d_clean)
            if (
                not c_nums
                or not d_nums
                or set(c_nums) == set(d_nums)
                or (len(c_nums) == 1 and c_nums[0] in d_nums)
                or (len(d_nums) == 1 and d_nums[0] in c_nums)
            ):
                return True

    return False


def sync_disk_files_to_csv(csv_path: Path, output_dir: Path, failed_csv_path: Path = DEFAULT_FAILED_CSV_PATH):
    """
    실행 시 가장 먼저 raw_pdfs 디렉토리 내 실제 파일들과
    documents.csv의 file_name들을 대조(유사 단어 포함 인정)하여 downloaded 열 및 failed.csv를 갱신합니다.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        return [], []

    # 실제 디스크에 존재하는 파일 목록 수집 (NFC 정규화 및 크기 > 0)
    existing_files_set = set()
    for p in output_dir.iterdir():
        if p.is_file() and p.stat().st_size > 0:
            existing_files_set.add(nfc(p.name))

    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    if "downloaded" not in fields:
        fields.append("downloaded")

    matched_count = 0
    missing_count = 0

    for r in rows:
        fn = nfc(r.get("file_name", ""))
        matched_disk_file = None

        # 1. 완전 일치 우선 확인
        if fn in existing_files_set or (output_dir / fn).exists():
            matched_disk_file = fn
        else:
            # 2. 유사 파일명 대조 (예: 130292099630937500_KIFVIP2013-10.pdf <-> KIFVIP2013-10.pdf)
            for df in existing_files_set:
                if is_filename_match(fn, df):
                    matched_disk_file = df
                    break

        if matched_disk_file:
            r["downloaded"] = "True"
            matched_count += 1
            if matched_disk_file != fn:
                print(f"  [유사 파일 매칭] CSV: '{fn}' <===> 폴더: '{matched_disk_file}' (인정: downloaded=True)")
                link_path = output_dir / fn
                if not link_path.exists():
                    try:
                        link_path.symlink_to(matched_disk_file)
                    except Exception:
                        pass
        else:
            r["downloaded"] = "False"
            missing_count += 1

    # documents.csv 즉시 저장
    with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # failed.csv 즉시 갱신
    save_failed_csv(rows, failed_csv_path)

    print(f"[사전 대조] raw_pdfs 파일 대조 완료:")
    print(f"- 실제 파일 존재 -> downloaded=True 반영: {matched_count}개")
    print(f"- 실제 파일 미존재 -> downloaded=False 반영: {missing_count}개")
    print(f"- 대조 결과 {csv_path}에 즉시 동기화 완료.\n")

    return fields, rows


def download_pdfs(
    csv_path: Path,
    output_dir: Path,
    failed_csv_path: Path = DEFAULT_FAILED_CSV_PATH,
    overwrite: bool = False,
    timeout: int = 30,
    insecure: bool = False,
):
    """
    documents.csv의 목록을 순회하며 PDF를 다운로드합니다.
    - 실행 전 sync_disk_files_to_csv()로 로컬 파일 유무를 먼저 대조하여 downloaded에 반영.
    - downloaded=True인 항목은 건너뛰고 (SKIP)
    - downloaded=False인 항목만 재시도 (RETRY)
    - 최종 완료 시 documents.csv 및 failed.csv를 갱신.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 가장 먼저 raw_pdfs 실제 파일 대조 및 downloaded/failed.csv 동기화
    fields, rows = sync_disk_files_to_csv(csv_path, output_dir, failed_csv_path=failed_csv_path)
    total = len(rows)

    already_done = sum(1 for r in rows if str(r.get("downloaded", "")).strip().lower() == "true")
    targets_to_run = total if overwrite else (total - already_done)

    print(f"[2단계] PDF 다운로드/재시도 시작")
    print(f"- 저장 디렉토리: {output_dir}")
    if not overwrite:
        print(f"- 다운로드 완료 항목 (스킵): {already_done}개")
        print(f"- 재시도 대상 (False): {targets_to_run}개\n")
    else:
        print(f"- --overwrite 옵션: 전체 무조건 재다운로드\n")

    if targets_to_run == 0 and not overwrite:
        print("모든 파일이 이미 다운로드되어 있습니다. 작업을 완료합니다.")
        return

    success_count = 0
    skipped_count = 0
    failed_items = []

    try:
        for idx, row in enumerate(rows, start=1):
            file_name = row.get("file_name", "").strip()
            url = row.get("url", "").strip()
            is_marked_true = str(row.get("downloaded", "")).strip().lower() == "true"

            if not file_name or not url:
                print(f"[{idx}/{total}] [건너뜀] 누락된 정보 (파일명: '{file_name}', URL: '{url}')")
                row["downloaded"] = "False"
                continue

            save_path = output_dir / file_name
            file_exists = save_path.exists() and save_path.stat().st_size > 0

            # 이미 True이고 파일이 존재하는 경우 스킵
            if is_marked_true and file_exists and not overwrite:
                print(f"[{idx}/{total}] [SKIP] 이미 다운로드 완료됨 (downloaded=True): {file_name}")
                skipped_count += 1
                continue

            # 다운로드 실행 (False인 항목)
            print(f"[{idx}/{total}] [다운로드/재시도] {file_name}")
            print(f"       URL: {url}")

            try:
                data, content_type, final_url = fetch_url(url, timeout=timeout, insecure=insecure, referer=url)
                is_pdf = data.startswith(b"%PDF")

                # 직접 응답이 PDF가 아니고 웹페이지(HTML)인 경우 내부 링크 추출 시도
                if not is_pdf and ("text/html" in content_type or data.lstrip().startswith(b"<!DOCTYPE") or data.lstrip().startswith(b"<html")):
                    print(f"       [정보] 웹 페이지 감지됨. PDF 첨부 링크 탐색 시도...")
                    try:
                        html_text = data.decode("utf-8", errors="ignore")
                        extracted_link = try_find_pdf_link_in_html(html_text, final_url)
                        if extracted_link and extracted_link != url:
                            print(f"       [재시도] 추출된 링크: {extracted_link}")
                            data, content_type, _ = fetch_url(extracted_link, timeout=timeout, insecure=insecure, referer=url)
                            is_pdf = data.startswith(b"%PDF")
                    except Exception as ex:
                        print(f"       [주의] 링크 파싱 오류: {ex}")

                if is_pdf:
                    with open(save_path, "wb") as out_f:
                        out_f.write(data)
                    size_kb = len(data) / 1024
                    print(f"       [성공] 저장 완료 ({size_kb:,.1f} KB)")
                    row["downloaded"] = "True"
                    success_count += 1
                else:
                    reason = "다운로드된 데이터가 PDF 형식이 아님 (게시판 본문 또는 보안 페이지 등 수동 다운로드 필요)"
                    print(f"       [실패] {reason}")
                    row["downloaded"] = "True" if (save_path.exists() and save_path.stat().st_size > 0) else "False"
                    failed_items.append((file_name, url, reason))

            except urllib.error.HTTPError as e:
                err_msg = f"HTTP Error {e.code}: {e.reason}"
                print(f"       [실패] {err_msg}")
                row["downloaded"] = "True" if (save_path.exists() and save_path.stat().st_size > 0) else "False"
                failed_items.append((file_name, url, err_msg))
            except urllib.error.URLError as e:
                err_msg = f"URL Error: {e.reason}"
                print(f"       [실패] {err_msg}")
                row["downloaded"] = "True" if (save_path.exists() and save_path.stat().st_size > 0) else "False"
                failed_items.append((file_name, url, err_msg))
            except Exception as e:
                err_msg = f"Error: {e}"
                print(f"       [실패] {err_msg}")
                row["downloaded"] = "True" if (save_path.exists() and save_path.stat().st_size > 0) else "False"
                failed_items.append((file_name, url, err_msg))

            time.sleep(0.3)

    finally:
        # 1. documents.csv 갱신 저장
        with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[기록 완료] documents.csv의 'downloaded' 열이 최종 갱신되었습니다.")

        # 2. failed.csv 최종 갱신 저장
        save_failed_csv(rows, failed_csv_path)

    print("\n" + "=" * 50)
    print("다운로드 결과 요약")
    print(f"- 전체: {total}개")
    print(f"- 기존 보유(건너뜀): {skipped_count}개")
    print(f"- 신규 성공: {success_count}개")
    print(f"- 최종 실패 / 수동 확인 필요: {len(failed_items)}개")
    print("=" * 50)

    if failed_items:
        print("\n[수동 확인 필요한 목록]")
        for fn, u, reason in failed_items:
            print(f"• {fn}\n  - 사유: {reason}\n  - URL: {u}\n")


def main():
    parser = argparse.ArgumentParser(description="documents.csv 및 PDF 일괄 다운로더")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"data 디렉토리 경로 (기본값: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"documents.csv 경로 (기본값: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--failed-csv",
        type=Path,
        default=DEFAULT_FAILED_CSV_PATH,
        help=f"실패 목록 failed.csv 저장 경로 (기본값: {DEFAULT_FAILED_CSV_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"PDF 저장 디렉토리 (기본값: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--force-csv",
        action="store_true",
        help="documents.csv가 이미 있어도 다시 다운로드",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="downloaded 열 유무 및 값과 무관하게 전체 무조건 재다운로드",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP 요청 타임아웃 초 단위 (기본값: 30)",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="SSL 인증서 검증 비활성화",
    )
    parser.add_argument(
        "-u",
        "--update",
        action="store_true",
        dest="update",
        help="다운로드 없이 raw_pdfs 폴더의 파일 목록만 확인하여 documents.csv의 downloaded 열과 failed.csv를 업데이트하고 종료",
    )
    args = parser.parse_args()

    # 1. documents.csv 다운로드 (없을 경우)
    download_documents_csv(args.data_dir, args.csv, force=args.force_csv)

    if args.update:
        sync_disk_files_to_csv(args.csv, args.output_dir, failed_csv_path=args.failed_csv)
        print("[완료] -u 모드: raw_pdfs 폴더 파일 대조 후 documents.csv 및 failed.csv 업데이트를 완료하고 종료합니다.")
        return

    # 2. 파일 대조 후 선별 다운로드 실행 및 failed.csv 생성
    download_pdfs(
        csv_path=args.csv,
        output_dir=args.output_dir,
        failed_csv_path=args.failed_csv,
        overwrite=args.overwrite,
        timeout=args.timeout,
        insecure=args.insecure,
    )


if __name__ == "__main__":
    main()
