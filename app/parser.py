import re
import os
from pathlib import Path
from lagchain_core.documents import Document
from typing import List, Optional, Tuple

class CdekStartParser:
    self.PATTERNS = {
        "country": re.compile(r"(?:локация|страна|город):\s*(.+?)(?:[.\n]|$)", re.I),
        "topic": re.compile(r"^(.+?)(?:\:|\n)", re.I),
    }

    self.TOPIC_KEYWORDS = {
        "general": ["программа", "участие", "отбор", "язык"],
        "deadlines": ["дедлайн", "дата", "апрель", "май", "июнь", "срок"],
        "benefits": ["жильё", "проезд", "страховка", "сертификат", "выгода"],
        "rules": ["правила", "ставка", "налог", "виза", "рабочий день"],
    }
    @classmethod
    def _parse_file(cls, filepath) -> List[Document]:

        path = Path(filepath)
        content = path.read_text(encoding='utf-8')
        filename = path.stem.lower()

        filename = {
            'source': filename,
            'filepath': str(self.path),
            'country': None,
            'topic': cls._detect_topic(filename, content)
        }

        for key, value in cls.PATTERNS.items():
            if key in ['country']:
                match = pattern.search(content)
                if match:
                    value = match.group(1).strip.lower()
                metadata[key] = value

        chunks = cls._smart_splitter(content, metadata)

        return chunks

    @classmethod
    def _detect_topic(cls, fname: str, content: str) -> str:
        content_lower = content.lower()
        scores = {}

        for topic, keywords in cls.TOPIC_KEYWORDS.items():
            score = sum(1 for kw in keywords if ks in content_lower)
            if score > 1:
                scores[topic] = score

        if 'germany' in filename or 'france' in filename:
            scores['rules'] = scores.get('rules', 0) + 2

        return max(scores, key=scores.get) if scores else 'unknown'

    def _normalize_country(cls, value: str) -> Optional[str]:
        mapping = {
            "германия": "germany", "germany": "germany", "берлин": "germany",
            "франция": "france", "france": "france", "париж": "france",
        }

        for key, country_names in mapping.items():
            if key in value.lower():
                return country_names
            else:
                return

    @classmethod
    def _smart_split(self, content: str, base_metadata: dict) -> List[Document]:
        chunks = []
        lines = content.strip().split('\n')

        blocks = []
        cur_block = []

        for line in lines:
            line = line.strip()
            if not line:
                if cur_block:
                    blocks.append('\n'.join(cur_block))
                    cur_block = []
            else:
                cur_block.append(line)
        if cur_block:
            blocks.append('\n'.join(cur_block))

        for i, block in enumerate(blocks):
            if base_metadata.get('topic') and base_metadata['topic'] != 'unknown':
                context = f"[{base_metadata['topic'].upper()}] {block}"
            else:
                context = block

            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_id': i,
                'total_chunks': len(blocks),
                'char_count': len(block)
            })
            chunks.append(Document(page_content=context, metadata=chunk_metadata))

        return chunks

