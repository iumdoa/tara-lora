"""Content invariants for source-grounded TARA-5 records (no API calls)."""
import hashlib
import re
from pathlib import Path
from src.schema_validator import validate_tara5_chapter

NOTE_PATTERN = re.compile(r'求月票|求推荐票|求订阅|求收藏|感谢[^\n]{0,90}打赏|(?:^|\n)\s*[Pp][Ss][:：]|感觉一直在单机有人不')

def validate_source_chapter(data, root: Path):
    ok, errors = validate_tara5_chapter(data)
    info=data.get('source_info',{})
    if info.get('boundary_policy') != 'disjoint_source_spans':
        return errors+['缺少可核验的原文切片策略']
    raw_path=(root/info.get('raw_path','')).resolve()
    if not raw_path.is_relative_to(root.resolve()/'data') or not raw_path.is_file():
        return errors+['原文路径不存在或超出数据目录']
    raw=raw_path.read_text(encoding='utf-8')
    if hashlib.sha256(raw.encode()).hexdigest()!=info.get('sha256'): errors.append('原文 SHA-256 不一致')
    # Dropping a sentence at the document edge is not allowed just because
    # an annotator chose a shorter range. Titles/author notes need justification.
    suffix=info.get('excluded_suffix','').strip()
    prefix=info.get('excluded_prefix','').strip()
    if suffix and re.search(r'[\u4e00-\u9fffA-Za-z0-9]',suffix):
        note=re.search(r'求.{0,5}票|打赏|加更|作者|[Pp][Ss][:：]|未完待续|本章完|全书完|感谢|盟主|更新|订阅|推荐|收藏|活动|小说|下载|电子书|网站|网址|网|首发|广告|书友|[a-zA-Z0-9.-]+\.(?:com|cn|net|org)|手打|游戏|大章节|百里|过生|月票|忘语|一更|构思|道友|爱尚|阅读|地址|顶点|动画|开机|开会|元旦|[。.]m\b',suffix)
        if not note: errors.append('尾部排除内容含叙事文字，缺少明确作话特征')
    if any(len(line.strip())>150 for line in prefix.splitlines()):
        errors.append('头部排除内容过长，疑似遗漏正文')
    last=None
    for stage,seg in data['segments'].items():
        lo,hi=seg.get('raw_char_offset_start'),seg.get('raw_char_offset_end')
        if type(lo) is not int or type(hi) is not int: continue
        if raw[lo:hi]!=seg['text']: errors.append(f'{stage}: 正文非原文精确切片')
        if last is not None and raw[last:lo].strip(): errors.append(f'{stage}: 阶段间遗漏非空白正文')
        last=hi
        if NOTE_PATTERN.search(seg['text']): errors.append(f'{stage}: 疑似作者作话，需要复核')
        for a in data['stage_plans'][stage]['action_chain']:
            spans=a.get('evidence_spans',[])
            if not spans: errors.append(f'{stage}: 行动缺少证据范围');continue
            quotes=[]
            for span in spans:
                start,end=span['start'],span['end']
                if not lo<=start<end<=hi: errors.append(f'{stage}: 行动证据不属于自身阶段')
                quotes.append(raw[start:end])
            if '\n'.join(quotes)!=a.get('evidence'): errors.append(f'{stage}: 行动证据与原文不一致')
    for fact in data.get('confirmed_facts',[]):
        source=(root/fact.get('source_path','')).resolve()
        if not source.is_relative_to(root.resolve()/'data') or not source.is_file():
            errors.append('事实前章来源不存在');continue
        if source==raw_path or source.parent!=raw_path.parent: errors.append('事实来源应为同书前章')
        current_index=re.search(r'ch_(\d+)',raw_path.name)
        previous_index=re.search(r'ch_(\d+)',source.name)
        if not current_index or not previous_index or int(previous_index[1])>=int(current_index[1]): errors.append('事实来源不是先前章节')
        if not fact.get('evidence') or fact['evidence'] not in source.read_text(): errors.append('事实引文未在前章出现')
        if fact.get('evidence') not in data.get('prev_chapter_ending',''): errors.append('事实不属于提供的前章上下文')
    return errors
