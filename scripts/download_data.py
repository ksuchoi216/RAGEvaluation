#!/usr/bin/env python3
"""
Hugging Face에서 documents.csv를 다운로드하고,
각 행의 url에서 PDF 파일을 다운로드하여 data/raw_pdfs/에 저장하는 스크립트.
"""

import argparse
import csv
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


def download_documents_csv(data_dir: Path, csv_path: Path, force: bool = False):
    """Hugging Face에서 documents.csv를 다운로드합니다."""
    data_dir.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() and not force:
        print(f"[1단계] documents.csv가 이미 존재합니다: {csv_path}")
        return

    print(f"[1단계] Hugging Face에서 documents.csv 다운로드 시작...")
    
    # 방법 1: hf CLI 사용 (사용자 환경에 hf 설치된 경우)
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
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
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

    # 방법 3: urllib 직접 다운로드 (HF resolve URL)
    print(f"       직접 HTTP 다운로드 시도: {HF_DIRECT_URL}")
    req = urllib.request.Request(HF_DIRECT_URL, headers=DEFAULT_HEADERS)
    ctx = create_ssl_context()
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        content = resp.read()
        with open(csv_path, "wb") as f:
            f.write(content)
    print(f"       HTTP 다운로드 완료: {csv_path} ({len(content):,} bytes)")


def fetch_url(url: str, timeout: int = 20, insecure: bool = False):
    """지정된 URL에서 바이너리 데이터를 다운로드합니다."""
    req = urllib.request.Request(url, headers=DEFAULT_HEADERS)
    ctx = create_ssl_context(insecure=insecure)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            data = response.read()
            final_url = response.geturl()
            return data, content_type, final_url
    except (ssl.SSLError, urllib.error.URLError) as e:
        if not insecure and ("certificate" in str(e).lower() or "ssl" in str(e).lower()):
            ctx_insecure = create_ssl_context(insecure=True)
            with urllib.request.urlopen(req, context=ctx_insecure, timeout=timeout) as response:
                content_type = response.headers.get("Content-Type", "").lower()
                data = response.read()
                final_url = response.geturl()
                return data, content_type, final_url
        raise e


def try_find_pdf_link_in_html(html_text: str, base_url: str):
    """HTML 페이지에서 PDF 첨부파일 다운로드 링크를 탐색합니다."""
    patterns = [
        r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']',
        r'href=["\']([^"\']*(?:download|fileDown|downDomainFile|file_down|attachment)[^"\']*)["\']',
    ]

    for pat in patterns:
        for match in re.finditer(pat, html_text, re.IGNORECASE):
            link = match.group(1).replace("&amp;", "&")
            if link.startswith("javascript:") or link.startswith("#"):
                continue
            return urllib.parse.urljoin(base_url, link)
    return None


def download_pdfs(
    csv_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    timeout: int = 20,
    insecure: bool = False,
):
    """documents.csv의 URL 목록을 순회하며 PDF를 다운로드합니다."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    print(f"\n[2단계] 총 {total}개의 PDF 파일 다운로드 시작")
    print(f"- 저장 디렉토리: {output_dir}\n")

    success_count = 0
    skipped_count = 0
    failed_items = []

    for idx, row in enumerate(rows, start=1):
        file_name = row.get("file_name", "").strip()
        url = row.get("url", "").strip()

        if not file_name or not url:
            print(f"[{idx}/{total}] [건너뜀] 누락된 정보 (파일명: '{file_name}', URL: '{url}')")
            continue

        save_path = output_dir / file_name

        if save_path.exists() and save_path.stat().st_size > 0 and not overwrite:
            print(f"[{idx}/{total}] [SKIP] 이미 존재함: {file_name}")
            skipped_count += 1
            continue

        print(f"[{idx}/{total}] 다운로드 중: {file_name}")
        print(f"       URL: {url}")

        try:
            data, content_type, final_url = fetch_url(url, timeout=timeout, insecure=insecure)
            is_pdf = data.startswith(b"%PDF")

            # 직접 응답이 PDF가 아니고 웹페이지(HTML)인 경우 내부 링크 추출 시도
            if not is_pdf and ("text/html" in content_type or data.lstrip().startswith(b"<!DOCTYPE") or data.lstrip().startswith(b"<html")):
                print(f"       [정보] 웹 페이지 감지됨. PDF 첨부 링크 탐색 시도...")
                try:
                    html_text = data.decode("utf-8", errors="ignore")
                    extracted_link = try_find_pdf_link_in_html(html_text, final_url)
                    if extracted_link and extracted_link != url:
                        print(f"       [재시도] 추출된 링크: {extracted_link}")
                        data, content_type, _ = fetch_url(extracted_link, timeout=timeout, insecure=insecure)
                        is_pdf = data.startswith(b"%PDF")
                except Exception as ex:
                    print(f"       [주의] 링크 파싱 오류: {ex}")

            if is_pdf:
                with open(save_path, "wb") as out_f:
                    out_f.write(data)
                size_kb = len(data) / 1024
                print(f"       [성공] 저장 완료 ({size_kb:,.1f} KB)")
                success_count += 1
            else:
                reason = "다운로드된 데이터가 PDF 형식이 아님 (게시판 본문 또는 보안 페이지 등 수동 다운로드 필요)"
                print(f"       [실패] {reason}")
                failed_items.append((file_name, url, reason))

        except urllib.error.HTTPError as e:
            err_msg = f"HTTP Error {e.code}: {e.reason}"
            print(f"       [실패] {err_msg}")
            failed_items.append((file_name, url, err_msg))
        except urllib.error.URLError as e:
            err_msg = f"URL Error: {e.reason}"
            print(f"       [실패] {err_msg}")
            failed_items.append((file_name, url, err_msg))
        except Exception as e:
            err_msg = f"Error: {e}"
            print(f"       [실패] {err_msg}")
            failed_items.append((file_name, url, err_msg))

        # 요청 간 간격
        time.sleep(0.3)

    print("\n" + "=" * 50)
    print("다운로드 결과 요약")
    print(f"- 전체: {total}개")
    print(f"- 성공: {success_count}개")
    print(f"- 건너뜀 (이미 존재): {skipped_count}개")
    print(f"- 실패 / 수동 확인 필요: {len(failed_items)}개")
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
        help="이미 존재하는 PDF 파일도 덮어쓰기",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="HTTP 요청 타임아웃 초 단위 (기본값: 20)",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="SSL 인증서 검증 비활성화",
    )
    args = parser.parse_args()

    # 1. documents.csv 다운로드
    download_documents_csv(args.data_dir, args.csv, force=args.force_csv)

    # 2. PDF 일괄 다운로드
    download_pdfs(
        csv_path=args.csv,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        timeout=args.timeout,
        insecure=args.insecure,
    )


if __name__ == "__main__":
    main()
