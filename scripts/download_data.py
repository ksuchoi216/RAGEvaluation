#!/usr/bin/env python3
"""
Hugging Face에서 documents.csv를 다운로드하고,
각 행의 url에서 PDF 파일을 다운로드하여 data/raw_pdfs/에 저장하며,
documents.csv에 'downloaded' 열(True/False)을 기록 및 갱신하는 스크립트.

재실행 시 downloaded=True 인 항목은 자동으로 건너뛰고, False인 항목만 재시도(retry)합니다.
"""

import argparse
import csv
import http.cookiejar
import os
import re
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# 기본 디렉토리 설정 (스크립트 위치 기준: ../data, ../data/raw_pdfs)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CSV_PATH = DEFAULT_DATA_DIR / "documents.csv"
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
        print(f"[1단계] documents.csv가 이미 존재합니다: {csv_path}")
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
    # 대법원(scourt) 등 직접 도메인 링크 우선 탐색
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


def sync_csv_status(csv_path: Path, output_dir: Path):
    """documents.csv에 파일 존재 여부를 바탕으로 downloaded 열을 기록/갱신합니다."""
    if not csv_path.exists():
        return

    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    if "downloaded" not in fields:
        fields.append("downloaded")

    for r in rows:
        fn = r.get("file_name", "").strip()
        save_path = output_dir / fn
        is_exist = save_path.exists() and save_path.stat().st_size > 0
        r["downloaded"] = "True" if is_exist else "False"

    with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def download_pdfs(
    csv_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    timeout: int = 30,
    insecure: bool = False,
):
    """
    documents.csv의 목록을 순회하며 PDF를 다운로드합니다.
    - downloaded=True 이고 로컬 파일이 존재하면 건너뜁니다 (skip).
    - downloaded=False 이거나 로컬 파일이 없는 항목만 재시도(retry)합니다.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    if "downloaded" not in fields:
        fields.append("downloaded")

    total = len(rows)

    # 사전 통계 계산
    already_done = 0
    for r in rows:
        fn = r.get("file_name", "").strip()
        sp = output_dir / fn
        is_true = str(r.get("downloaded", "")).strip().lower() == "true"
        if (is_true or (sp.exists() and sp.stat().st_size > 0)) and not overwrite:
            already_done += 1

    targets_to_run = total if overwrite else (total - already_done)

    print(f"\n[2단계] 총 {total}개 문서 중 {targets_to_run}개 대상 처리 시작")
    print(f"- 저장 디렉토리: {output_dir}")
    if not overwrite:
        print(f"- 이미 다운로드 완료 (건너뜀): {already_done}개")
        print(f"- 다운로드/재시도(Retry) 대상: {targets_to_run}개\n")
    else:
        print(f"- --overwrite 옵션 활성화: 전체 재다운로드 진행\n")

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

            # 1. 이미 다운로드 완료(True)이고 파일이 실제 존재하는 경우 -> SKIP
            if is_marked_true and file_exists and not overwrite:
                print(f"[{idx}/{total}] [SKIP] 이미 다운로드 완료됨 (downloaded=True): {file_name}")
                skipped_count += 1
                continue

            # 2. downloaded는 False였으나 파일이 이미 디스크에 존재하는 경우 -> 상태를 True로 승격하고 SKIP
            if file_exists and not overwrite:
                print(f"[{idx}/{total}] [SKIP] 로컬 파일 감지됨 -> downloaded=True 갱신: {file_name}")
                row["downloaded"] = "True"
                skipped_count += 1
                continue

            # 3. 다운로드 시도 (Retry 대상)
            retry_reason = "[재시도 대상(downloaded=False)]" if not is_marked_true else "[로컬 파일 누락으로 재다운로드]"
            print(f"[{idx}/{total}] {retry_reason} 다운로드 중: {file_name}")
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

            # 요청 간 간격
            time.sleep(0.3)

    finally:
        # 다운로드 결과를 documents.csv의 downloaded 열에 저장
        with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[기록 완료] documents.csv의 'downloaded' 열이 최신 상태로 갱신되었습니다.")

    print("\n" + "=" * 50)
    print("다운로드 결과 요약")
    print(f"- 전체: {total}개")
    print(f"- 기존 완료(건너뜀): {skipped_count}개")
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
        help="downloaded=True 및 로컬 파일 유무와 관계없이 전체 강제 재다운로드",
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
        "--sync-only",
        action="store_true",
        help="다운로드 없이 현재 raw_pdfs 파일 유무만 documents.csv에 동기화",
    )
    args = parser.parse_args()

    # documents.csv 다운로드
    download_documents_csv(args.data_dir, args.csv, force=args.force_csv)

    if args.sync_only:
        sync_csv_status(args.csv, args.output_dir)
        print(f"[완료] {args.output_dir}의 파일 목록을 바탕으로 {args.csv}의 downloaded 열을 동기화했습니다.")
        return

    # PDF 일괄 다운로드 및 status 기록 (True는 스킵, False만 retry)
    download_pdfs(
        csv_path=args.csv,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        timeout=args.timeout,
        insecure=args.insecure,
    )


if __name__ == "__main__":
    main()
