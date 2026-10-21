#!/usr/bin/env python3
"""
TARA 小说分章工具 (Novel Splitter)
支持格式：
1. .epub 电子书（基于标准库 zipfile + xml/html 解析，保留天然目录结构与纯文本）
2. .txt 文本文件（基于智能正则识别章节标题）

输出：
将章节按规范提取并保存到 data/raw/ 目录，供 segmenter.py 进一步执行 TARA 逆向切分。
"""

import os
import re
import sys
import zipfile
import argparse
from pathlib import Path
from html.parser import HTMLParser
import xml.etree.ElementTree as ET

CURRENT_DIR = Path(__file__).resolve().parent
LORA_ROOT = CURRENT_DIR.parent
DEFAULT_RAW_DIR = LORA_ROOT / "data" / "raw"


class HTMLTextExtractor(HTMLParser):
    """纯标准库实现的轻量 HTML 正文提取器，保留段落换行"""
    def __init__(self, in_body: bool = True):
        super().__init__()
        self.text_parts = []
        self.in_body = in_body
        self.ignore_tags = {"script", "style", "head", "meta"}
        self.current_ignore = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "body":
            self.in_body = True
        if tag in self.ignore_tags:
            self.current_ignore = True
        if tag in {"p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "li"}:
            self.text_parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self.ignore_tags:
            self.current_ignore = False
        if tag in {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "li"}:
            self.text_parts.append("\n")

    def handle_data(self, data):
        if not self.current_ignore:
            self.text_parts.append(data)

    def get_text(self) -> str:
        raw_text = "".join(self.text_parts)
        # 规整化连续空行和首尾空格
        lines = [line.strip() for line in raw_text.splitlines()]
        cleaned_lines = [l for l in lines if l]
        return "\n\n".join(cleaned_lines)


def sanitize_filename(name: str) -> str:
    """清理文件名中的非法字符"""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    return name.strip()[:60]


def extract_chapters_from_epub(epub_path: Path, output_dir: Path) -> list:
    """从 EPUB 文件中提取所有章节（优先解析 toc.ncx 保证卷名与完整章名）"""
    print(f"[*] 正在解析 EPUB 文件: {epub_path}")
    if not zipfile.is_zipfile(epub_path):
        raise ValueError(f"不是有效的 ZIP/EPUB 文件: {epub_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_files = []

    with zipfile.ZipFile(epub_path, "r") as zf:
        namelist = zf.namelist()
        
        # 1. 查找 container.xml 获取 rootfile (OPF) 路径
        container_path = "META-INF/container.xml"
        if container_path not in namelist:
            raise ValueError("EPUB 缺少 META-INF/container.xml")

        container_xml = zf.read(container_path).decode("utf-8")
        root = ET.fromstring(container_xml)
        rootfile_elem = root.find(".//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile")
        if rootfile_elem is None:
            raise ValueError("未找到 rootfile 节点")
        opf_path = rootfile_elem.attrib.get("full-path", "")
        opf_dir = str(Path(opf_path).parent)
        if opf_dir == ".":
            opf_dir = ""

        # 2. 读取 OPF 解析 manifest 和 spine
        opf_content = zf.read(opf_path).decode("utf-8")
        opf_root = ET.fromstring(opf_content)
        
        manifest = {}
        for item in opf_root.findall(".//{http://www.idpf.org/2007/opf}item") or opf_root.findall(".//item"):
            manifest[item.attrib.get("id")] = item.attrib.get("href")

        spine_ids = []
        for itemref in opf_root.findall(".//{http://www.idpf.org/2007/opf}itemref") or opf_root.findall(".//itemref"):
            idref = itemref.attrib.get("idref")
            if idref:
                spine_ids.append(idref)

        # 3. 尝试解析 toc.ncx 获取权威目录结构（分卷与完整章名）
        nav_entries = []
        has_multi_chapters_in_file = False
        ncx_candidates = [n for n in namelist if n.endswith(".ncx")]
        if ncx_candidates:
            try:
                ncx_xml = zf.read(ncx_candidates[0]).decode("utf-8", errors="ignore")
                ncx_root = ET.fromstring(ncx_xml)
                current_vol = ""
                file_counter = {}
                for nav in ncx_root.findall(".//{http://www.daisy.org/z3986/2005/ncx/}navPoint"):
                    t_elem = nav.find(".//{http://www.daisy.org/z3986/2005/ncx/}text")
                    c_elem = nav.find(".//{http://www.daisy.org/z3986/2005/ncx/}content")
                    txt = t_elem.text.strip() if t_elem is not None and t_elem.text else ""
                    src = c_elem.attrib.get("src", "") if c_elem is not None else ""
                    src_clean = Path(src.split("#")[0]).name
                    anchor_part = src.split("#")[1] if "#" in src else ""
                    
                    if ("卷" in txt or "部" in txt) and not ("章" in txt or "节" in txt) and not ("总结" in txt):
                        current_vol = txt
                    else:
                        nav_entries.append({
                            "vol": current_vol,
                            "title": txt,
                            "src_file": src_clean,
                            "anchor": anchor_part
                        })
                        file_counter[src_clean] = file_counter.get(src_clean, 0) + 1
                        if anchor_part or file_counter[src_clean] > 1:
                            has_multi_chapters_in_file = True
                print(f"[*] 成功从 {ncx_candidates[0]} 解析到 {len(nav_entries)} 条目录索引")
            except Exception as e:
                print(f"[!] 解析 NCX 目录失败: {e}，将回退至 HTML 首行推断")

        # 模式 A：当电子书采用单个 HTML 内嵌多章或锚点切分 (如 Kindle/Calibre 转化本)
        if has_multi_chapters_in_file and len(nav_entries) > len(spine_ids):
            print(f"[*] 检测到单 HTML 聚合多章模式 (共 {len(nav_entries)} 章，分布于多个分块页面)，启用锚点切分流水线...")
            grouped = {}
            for entry in nav_entries:
                grouped.setdefault(entry["src_file"], []).append(entry)

            chapter_idx = 1
            for src_file, entries in grouped.items():
                matches = [name for name in namelist if name.endswith(src_file)]
                if not matches:
                    continue
                html_data = zf.read(matches[0]).decode("utf-8", errors="ignore")

                # 定位本文件内各章节锚点或标题起始位置
                entry_positions = []
                for e in entries:
                    anc = e["anchor"]
                    pos = -1
                    if anc:
                        pos = html_data.find('"' + anc + '"')
                        if pos == -1:
                            pos = html_data.find("'" + anc + "'")
                    if pos == -1:
                        pos = html_data.find(e["title"])
                    entry_positions.append((pos, e))

                for i, (pos, e) in enumerate(entry_positions):
                    if pos == -1:
                        continue
                    next_pos = len(html_data)
                    for j in range(i + 1, len(entry_positions)):
                        if entry_positions[j][0] != -1:
                            next_pos = entry_positions[j][0]
                            break
                    chunk = html_data[pos:next_pos]
                    parser = HTMLTextExtractor()
                    parser.feed(chunk)
                    chapter_text = parser.get_text()

                    # 清理锚点残留
                    anc = e["anchor"]
                    prefix = 'id="' + anc + '">'
                    if anc and chapter_text.startswith(prefix):
                        chapter_text = chapter_text[len(prefix):].strip()

                    title_candidate = f"{e['vol']}_{e['title']}" if e["vol"] else e["title"]
                    if not chapter_text.startswith(title_candidate) and not chapter_text.startswith(e["title"]):
                        chapter_text = f"{title_candidate}\n\n{chapter_text}"

                    if len(chapter_text) < 150:
                        continue

                    safe_title = sanitize_filename(title_candidate)
                    filename = f"ch_{chapter_idx:04d}_{safe_title}.txt"
                    target_path = output_dir / filename

                    with open(target_path, "w", encoding="utf-8") as out_f:
                        out_f.write(chapter_text)

                    extracted_files.append((chapter_idx, safe_title, len(chapter_text), target_path))
                    chapter_idx += 1

            return extracted_files

        # 模式 B：标准 1-HTML-1-Chapter 模式 (凡人修仙传、诡秘之主、大奉打更人等)
        print(f"[*] 发现 {len(spine_ids)} 个阅读顺序片段，开始提取正文...")
        toc_map = {e["src_file"]: (e["vol"], e["title"]) for e in nav_entries}
        chapter_idx = 1
        for idref in spine_ids:
            href = manifest.get(idref)
            if not href:
                continue

            full_href = f"{opf_dir}/{href}".lstrip("/") if opf_dir else href.lstrip("/")
            full_href = str(Path(full_href))
            if full_href not in namelist:
                matches = [name for name in namelist if name.endswith(Path(href).name)]
                if matches:
                    full_href = matches[0]
                else:
                    continue

            html_data = zf.read(full_href).decode("utf-8", errors="ignore")
            
            parser = HTMLTextExtractor()
            parser.feed(html_data)
            chapter_text = parser.get_text()

            # 过滤过短页面（如封面、版权等）
            if len(chapter_text) < 150:
                continue

            src_key = Path(full_href).name
            if src_key in toc_map:
                vol, title = toc_map[src_key]
                if vol:
                    title_candidate = f"{vol}_{title}"
                else:
                    title_candidate = title
            else:
                first_lines = chapter_text.splitlines()
                title_candidate = first_lines[0] if first_lines else f"第{chapter_idx}章"
                if len(title_candidate) > 40:
                    title_candidate = f"第{chapter_idx}章"

            safe_title = sanitize_filename(title_candidate)
            filename = f"ch_{chapter_idx:04d}_{safe_title}.txt"
            target_path = output_dir / filename

            # 确保正文开头包含完整标题，方便后续提炼
            if not chapter_text.startswith(title_candidate):
                chapter_text = f"{title_candidate}\n\n{chapter_text}"

            with open(target_path, "w", encoding="utf-8") as out_f:
                out_f.write(chapter_text)

            extracted_files.append((chapter_idx, safe_title, len(chapter_text), target_path))
            chapter_idx += 1

        return extracted_files


def extract_chapters_from_txt(txt_path: Path, output_dir: Path) -> list:
    """从单本 TXT 纯文本小说中正则切分章节"""
    print(f"[*] 正在按正则切分 TXT 小说: {txt_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # 常用网文章节标题正则
    pattern = re.compile(
        r'(^\s*(?:第[0-9一二三四五六七八九十百千万]+[章回卷节篇集]|Chapter\s*\d+|[0-9]{1,4}\s+).*$)',
        re.MULTILINE
    )

    matches = list(pattern.finditer(content))
    if not matches:
        # 如果未匹配到明确章节名，尝试按字数分块（每 2500 字一块）
        print("[!] 未检测到标准章节标题正则，按 2500 字定长切片模式兜底...")
        chunk_size = 2500
        extracted_files = []
        for idx, i in enumerate(range(0, len(content), chunk_size), 1):
            chunk = content[i : i + chunk_size].strip()
            filename = f"ch_{idx:04d}_片段.txt"
            target_path = output_dir / filename
            with open(target_path, "w", encoding="utf-8") as out_f:
                out_f.write(chunk)
            extracted_files.append((idx, f"片段_{idx}", len(chunk), target_path))
        return extracted_files

    extracted_files = []
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start_pos = match.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        
        chapter_body = content[start_pos:end_pos].strip()
        if len(chapter_body) < 100:
            continue

        chapter_idx = i + 1
        safe_title = sanitize_filename(title)
        filename = f"ch_{chapter_idx:04d}_{safe_title}.txt"
        target_path = output_dir / filename

        with open(target_path, "w", encoding="utf-8") as out_f:
            out_f.write(chapter_body)

        extracted_files.append((chapter_idx, safe_title, len(chapter_body), target_path))

    return extracted_files


def main():
    parser = argparse.ArgumentParser(description="TARA 小说分章工具 (支持 EPUB / TXT)")
    parser.add_argument("file", type=str, help="待拆解的小说文件路径 (.epub 或 .txt)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_RAW_DIR), help="章节输出目录，默认为 data/raw/")
    args = parser.parse_args()

    input_path = Path(args.file)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"[-] 错误：文件不存在 {input_path}")
        sys.exit(1)

    suffix = input_path.suffix.lower()
    if suffix == ".epub":
        results = extract_chapters_from_epub(input_path, output_dir)
    elif suffix == ".txt":
        results = extract_chapters_from_txt(input_path, output_dir)
    else:
        print(f"[-] 不支持的文件后缀: {suffix}。仅支持 .epub 和 .txt 格式。")
        sys.exit(1)

    print("\n" + "=" * 50)
    print(f"【分章完成】共成功提取 {len(results)} 章到 {output_dir}")
    print("样例预览（前 5 章）：")
    for idx, title, length, path in results[:5]:
        print(f"  - [{idx:03d}] {title} ({length} 字) -> {path.name}")
    if len(results) > 5:
        print(f"  ... 以及其余 {len(results) - 5} 章")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
