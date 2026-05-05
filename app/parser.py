import re
import os
from pathlib import Path
from langchain_core.documents import Document
from typing import List, Optional, Dict, Any


class CdekStartParser:
    """Парсер документов для системы стажировки CdekStart."""
    
    PATTERNS: Dict[str, re.Pattern] = {
        "country": re.compile(r"(?:локация|страна|город):\s*(.+?)(?:[.\n]|$)", re.I),
        "topic": re.compile(r"^(.+?)(?:\:|\n)", re.I),
    }

    TOPIC_KEYWORDS: Dict[str, List[str]] = {
        "general": ["программа", "участие", "отбор", "язык"],
        "deadlines": ["дедлайн", "дата", "апрель", "май", "июнь", "срок"],
        "benefits": ["жильё", "проезд", "страховка", "сертификат", "выгода"],
        "rules": ["правила", "ставка", "налог", "виза", "рабочий день"],
    }

    COUNTRY_MAPPING: Dict[str, str] = {
        "германия": "germany", 
        "germany": "germany", 
        "берлин": "germany",
        "франция": "france", 
        "france": "france", 
        "париж": "france",
        "berlin": "germany",
        "paris": "france",
    }

    def __init__(self, data_dir: str = "data/"):
        """Инициализация парсера.
        
        Args:
            data_dir: Путь к директории с данными.
        """
        self.data_dir = Path(data_dir)

    def parse_all(self) -> List[Document]:
        """Парсит все файлы в директории данных.
        
        Returns:
            Список документов LangChain.
        """
        all_documents = []
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Директория данных не найдена: {self.data_dir}")
        
        for filepath in self.data_dir.glob("*.txt"):
            try:
                docs = self.parse_file(filepath)
                all_documents.extend(docs)
            except Exception as e:
                print(f"Ошибка при парсинге {filepath}: {e}")
        
        return all_documents

    def parse_file(self, filepath: Path | str) -> List[Document]:
        """Парсит один файл.
        
        Args:
            filepath: Путь к файлу.
            
        Returns:
            Список документов LangChain.
        """
        path = Path(filepath)
        content = path.read_text(encoding='utf-8')
        filename = path.stem.lower()

        metadata: Dict[str, Any] = {
            'source': filename,
            'filepath': str(path),
            'country': self._detect_country_from_filename(filename),
            'topic': self._detect_topic(filename, content)
        }

        # Извлекаем страну из контента, если есть паттерн
        country_match = self.PATTERNS["country"].search(content)
        if country_match:
            country_value = country_match.group(1).strip().lower()
            normalized_country = self._normalize_country(country_value)
            if normalized_country:
                metadata['country'] = normalized_country

        chunks = self._smart_splitter(content, metadata)
        return chunks

    def _detect_country_from_filename(self, filename: str) -> Optional[str]:
        """Определяет страну по имени файла.
        
        Args:
            filename: Имя файла без расширения.
            
        Returns:
            Нормализованное название страны или None.
        """
        filename_lower = filename.lower()
        
        for key, country in self.COUNTRY_MAPPING.items():
            if key in filename_lower:
                return country
        
        return None

    def _detect_topic(self, fname: str, content: str) -> str:
        """Определяет тему документа по ключевым словам.
        
        Args:
            fname: Имя файла.
            content: Содержимое файла.
            
        Returns:
            Название темы.
        """
        content_lower = content.lower()
        scores: Dict[str, int] = {}

        for topic, keywords in self.TOPIC_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > 0:
                scores[topic] = score

        # Бонус для правил по имени файла
        if 'germany' in fname or 'france' in fname:
            scores['rules'] = scores.get('rules', 0) + 2

        return max(scores, key=scores.get) if scores else 'unknown'

    def _normalize_country(self, value: str) -> Optional[str]:
        """Нормализует название страны.
        
        Args:
            value: Строка с названием страны.
            
        Returns:
            Нормализованное название страны или None.
        """
        value_lower = value.lower()
        
        for key, country in self.COUNTRY_MAPPING.items():
            if key in value_lower:
                return country
        
        return None

    def _smart_splitter(self, content: str, base_metadata: Dict[str, Any]) -> List[Document]:
        """Разбивает контент на смысловые блоки.
        
        Args:
            content: Текст документа.
            base_metadata: Базовые метаданные.
            
        Returns:
            Список документов LangChain.
        """
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
