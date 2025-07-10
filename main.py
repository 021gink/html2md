import os
import time
import threading
import re
import asyncio
import logging
import json
import queue
import sys
import httpx
from enum import Enum
from pathlib import Path
from functools import wraps
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List, Tuple

import bs4
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler
from readabilipy import simple_json_from_html_string
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import requests
import html_to_markdown
from markdowncleaner import MarkdownCleaner, CleanerOptions

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("converter.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def error_handler(logger_instance: logging.Logger, default_return=None):
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger_instance.error(f"Async error in {func.__name__}: {e}", exc_info=True)
                return default_return if default_return is not None else {'error': str(e), 'success': False}
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger_instance.error(f"Sync error in {func.__name__}: {e}", exc_info=True)
                return default_return if default_return is not None else {'error': str(e), 'success': False}
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

class ConfigKeys(Enum):
    TIMEOUT = "timeout"
    QUALITY_THRESHOLD = "quality_threshold"
    AD_THRESHOLD = "ad_threshold"
    PARSER = "parser"
    MAX_FILE_SIZE_MB = "max_file_size_mb"
    AD_PATTERNS = "ad_patterns"
    PRIMARY_CONVERTER = "primary_converter"
    ENABLE_MARKDOWN_CLEANING = "enable_markdown_cleaning"
    MIN_LINE_LENGTH = "min_line_length"
    REMOVE_SHORT_LINES = "remove_short_lines"
    REMOVE_FOOTNOTES = "remove_footnotes"
    REMOVE_DUPLICATE_HEADLINES = "remove_duplicate_headlines"
    CONTRACT_EMPTY_LINES = "contract_empty_lines"
    CRIMP_LINEBREAKS = "crimp_linebreaks"
    PRESERVE_CODE_BLOCKS = "preserve_code_blocks"
    REMOVE_IMAGES = "remove_images"
    REMOVE_LINKS = "remove_links"

class ConfigManager:
    def __init__(self, config_path: str = "converter_config.json"):
        self.config_path = Path(config_path)
        self.config = self._get_default_config()
        self.load()
    def _get_default_config(self):
        return {
            ConfigKeys.TIMEOUT: 30,
            ConfigKeys.QUALITY_THRESHOLD: 90.0,
            ConfigKeys.AD_THRESHOLD: 3.0,
            ConfigKeys.PARSER: "lxml",
            ConfigKeys.MAX_FILE_SIZE_MB: 10,
            ConfigKeys.AD_PATTERNS: [
                '广告', '赞助', 'promoted', 'sponsored', '推荐阅读', '相关文章',
                '你可能喜欢', '关注我们', '订阅', 'newsletter',
            ],
            ConfigKeys.PRIMARY_CONVERTER: "readability",
            ConfigKeys.ENABLE_MARKDOWN_CLEANING: True,
            ConfigKeys.MIN_LINE_LENGTH: 50,
            ConfigKeys.REMOVE_SHORT_LINES: True,
            ConfigKeys.REMOVE_FOOTNOTES: True,
            ConfigKeys.REMOVE_DUPLICATE_HEADLINES: True,
            ConfigKeys.CONTRACT_EMPTY_LINES: True,
            ConfigKeys.CRIMP_LINEBREAKS: True,
            ConfigKeys.PRESERVE_CODE_BLOCKS: True,
            ConfigKeys.REMOVE_IMAGES: False,
            ConfigKeys.REMOVE_LINKS: False,
        }
    @error_handler(logger)
    def load(self):
        if self.config_path.exists():
            with self.config_path.open('r', encoding='utf-8') as f:
                user_config = json.load(f)
                for key_str, value in user_config.items():
                    try:
                        key = ConfigKeys(key_str)
                        self.config[key] = value
                    except ValueError:
                        logger.warning(f"Unknown config key: {key_str}, ignored.")
            logger.info(f"Config loaded from {self.config_path}.")
    @error_handler(logger)
    def save(self):
        save_config = {key.value: value for key, value in self.config.items()}
        with self.config_path.open('w', encoding='utf-8') as f:
            json.dump(save_config, f, indent=2, ensure_ascii=False)
        logger.info(f"Config saved to {self.config_path}.")
    def get(self, key: ConfigKeys, default=None):
        return self.config.get(key, default)
    @error_handler(logger)
    def set(self, key: ConfigKeys, value: Any):
        self.config[key] = value
        self.save()

class MarkdownPostProcessor:
    def __init__(self):
        self.cleaner = None
        self._initialize_cleaner()
    def _initialize_cleaner(self):
        try:
            options = CleanerOptions()
            options.remove_short_lines = config_manager.get(ConfigKeys.REMOVE_SHORT_LINES, True)
            options.min_line_length = config_manager.get(ConfigKeys.MIN_LINE_LENGTH, 50)
            options.remove_footnotes_in_text = config_manager.get(ConfigKeys.REMOVE_FOOTNOTES, True)
            options.remove_duplicate_headlines = config_manager.get(ConfigKeys.REMOVE_DUPLICATE_HEADLINES, True)
            options.contract_empty_lines = config_manager.get(ConfigKeys.CONTRACT_EMPTY_LINES, True)
            options.crimp_linebreaks = config_manager.get(ConfigKeys.CRIMP_LINEBREAKS, True)
            self.cleaner = MarkdownCleaner(options=options)
            logger.info("MarkdownCleaner initialized.")
        except Exception as e:
            logger.warning(f"MarkdownCleaner init failed: {e}")
            self.cleaner = None
    def refresh_cleaner(self):
        self._initialize_cleaner()
    @error_handler(logger)
    def clean_markdown(self, markdown_text: str) -> str:
        if not config_manager.get(ConfigKeys.ENABLE_MARKDOWN_CLEANING, True):
            return markdown_text
        if not markdown_text or not markdown_text.strip():
            return markdown_text
        self._initialize_cleaner()
        if not self.cleaner:
            logger.warning("Cleaner not initialized. Returning original.")
            return markdown_text
        try:
            cleaned_text = self.cleaner.clean_markdown_string(markdown_text)
            logger.info(f"Markdown cleaned: {len(markdown_text)} -> {len(cleaned_text)}")
            return cleaned_text
        except Exception as e:
            logger.error(f"Markdown clean failed: {e}")
            return markdown_text

class InputTypeDetector:
    @staticmethod
    def detect(content: str) -> Tuple[str, str]:
        content = content.strip()
        if re.match(r'^https?://', content, re.IGNORECASE):
            return "url", content
        path = Path(content)
        if path.exists() and path.is_file():
            return "file", str(path)
        if content.startswith('<') and re.search(r'<[a-z][\s\S]*>', content, re.I):
            return "html", content
        if '.' in content and ' ' not in content:
            return "url", f'https://{content}'
        return "text", content

class HTMLProcessor:
    def __init__(self, html: str, parser: str):
        self.html = html
        self.parser = parser
        self.soup = None
    def __enter__(self):
        try:
            self.soup = BeautifulSoup(self.html, self.parser)
            return self.soup
        except Exception as e:
            logger.error(f"BeautifulSoup parse failed: {e}")
            return None
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.soup:
            self.soup.decompose()
        self.soup = None

class ConverterEngine(ABC):
    @abstractmethod
    async def convert(self, content: str, **kwargs) -> Dict[str, Any]:
        pass

class Crawl4AIConverter(ConverterEngine):
    @error_handler(logger, default_return={'error': 'Crawl4AI conversion failed'})
    async def convert(self, content: str, **kwargs):
        logger.info("Using Crawl4AIConverter...")
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(content)
            return {
                'html': result.html,
                'markdown': result.markdown,
                'cleaned_html': result.cleaned_html,
                'success': result.success,
                'status_code': result.status_code,
                'url': result.url,
                'title': result.metadata.get('title', ''),
                'author': result.metadata.get('author', '')
            }

class ReadabilityConverter(ConverterEngine):
    def _extract_text_from_soup(self, soup: BeautifulSoup) -> Tuple[str, str]:
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            element.decompose()
        title_tag = soup.find('title') or soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else "无标题"
        main_content = soup.find('article') or soup.find('main') or soup.body
        if not main_content:
            return title, "无法从 HTML 中提取有效内容。"
        markdown_parts = [f"# {title}"]
        for tag in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'pre']):
            text = tag.get_text(strip=True)
            if not text:
                continue
            if tag.name.startswith('h'):
                level = int(tag.name[1])
                markdown_parts.append(f"{'#' * level} {text}")
            elif tag.name == 'p':
                markdown_parts.append(text)
            elif tag.name == 'li':
                markdown_parts.append(f"- {text}")
            elif tag.name == 'pre':
                markdown_parts.append(f"```\n{text}\n```")
        return title, '\n\n'.join(markdown_parts)
    @error_handler(logger, default_return={'error': 'Readability conversion failed'})
    async def convert(self, html_content: str, **kwargs):
        logger.info("Using ReadabilityConverter...")
        try:
            article = simple_json_from_html_string(html_content, use_readability=True)
            if article and article.get('plain_text'):
                logger.info("readabilipy extraction succeeded.")
                title = article.get('title', '无标题')
                author = article.get('byline')
                plain_text = '\n\n'.join(p['text'] for p in article['plain_text'] if p.get('text'))
                markdown = f"# {title}\n\n"
                if author:
                    markdown += f"*作者: {author}*\n\n"
                markdown += plain_text
                return {
                    'html': html_content,
                    'markdown': markdown,
                    'cleaned_html': article.get('content', ''),
                    'success': True,
                    'title': title,
                    'author': author,
                }
        except Exception as e:
            logger.warning(f"readabilipy failed: {e}, fallback to soup.")
        logger.info("Fallback to BeautifulSoup extraction.")
        parser = config_manager.get(ConfigKeys.PARSER)
        with HTMLProcessor(html_content, parser) as soup:
            if not soup:
                raise ValueError("HTML parse failed.")
            title, markdown = self._extract_text_from_soup(soup)
            return {
                'html': html_content,
                'markdown': markdown,
                'cleaned_html': str(soup),
                'success': True,
                'title': title,
                'author': ''
            }

class JinaConverter(ConverterEngine):
    @error_handler(logger, default_return={'error': 'Jina conversion failed'})
    async def convert(self, content: str, **kwargs):
        logger.info("Using JinaConverter...")
        if not content.strip().lower().startswith("http"):
            return {'error': 'Jina only supports URL input', 'success': False}
        proxy_url = f"https://r.jina.ai/{content}"
        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, lambda: requests.get(proxy_url, timeout=30))
            resp.raise_for_status()
            markdown = resp.text
            return {
                'html': '',
                'markdown': markdown,
                'cleaned_html': '',
                'success': True,
                'status_code': resp.status_code,
                'url': content,
                'title': '',
                'author': '',
            }
        except Exception as e:
            logger.error(f"Jina conversion failed: {e}")
            return {'error': str(e), 'success': False}

class HtmlToMarkdownConverter(ConverterEngine):
    @error_handler(logger, default_return={'error': 'html-to-markdown conversion failed'})
    async def convert(self, content: str, **kwargs):
        logger.info("Using HtmlToMarkdownConverter...")
        try:
            markdown = html_to_markdown.convert_to_markdown(content)
            return {
                'html': content,
                'markdown': markdown,
                'cleaned_html': content,
                'success': True,
                'title': '',
                'author': '',
            }
        except Exception as e:
            logger.error(f"html-to-markdown failed: {e}")
            return {'error': str(e), 'success': False}

@dataclass
class ConversionMetrics:
    fidelity: float = 0.0
    ad_retention: float = 0.0
    word_count: int = 0
    elapsed_time: float = 0.0

class QualityAnalyzer:
    def __init__(self):
        self.ad_regex_cache = {}
    @error_handler(logger, default_return=ConversionMetrics())
    def calculate(self, original_html: str, result: Dict[str, Any], elapsed: float) -> ConversionMetrics:
        markdown = result.get('markdown', '')
        cleaned_html = result.get('cleaned_html', '')
        ad_retention = self._calculate_ad_retention(markdown)
        fidelity = self._calculate_fidelity(original_html, cleaned_html)
        word_count = len(markdown.split())
        return ConversionMetrics(
            fidelity=fidelity,
            ad_retention=ad_retention,
            word_count=word_count,
            elapsed_time=elapsed
        )
    def _calculate_ad_retention(self, markdown: str) -> float:
        ad_patterns = config_manager.get(ConfigKeys.AD_PATTERNS)
        if not ad_patterns or not markdown:
            return 0.0
        patterns_key = tuple(ad_patterns)
        if patterns_key not in self.ad_regex_cache:
            regex_str = "|".join(ad_patterns)
            self.ad_regex_cache[patterns_key] = re.compile(regex_str, re.IGNORECASE)
        regex = self.ad_regex_cache[patterns_key]
        matches = regex.findall(markdown)
        return (len(matches) / len(markdown.split())) * 100 if markdown.split() else 0.0
    def _calculate_fidelity(self, original: str, cleaned: str) -> float:
        parser = config_manager.get(ConfigKeys.PARSER)
        with HTMLProcessor(original, parser) as orig_soup, \
             HTMLProcessor(cleaned, parser) as clean_soup:
            if not orig_soup or not clean_soup:
                return 0.0
            elements_to_check = {
                'headers': (['h1', 'h2', 'h3', 'h4', 'h5', 'h6'], 0.4),
                'lists': (['ul', 'ol'], 0.3),
                'tables': (['table'], 0.2),
                'code': (['pre', 'code'], 0.1)
            }
            fidelity_score = 0.0
            for tags, weight in elements_to_check.values():
                orig_count = len(orig_soup.find_all(tags))
                clean_count = len(clean_soup.find_all(tags))
                ratio = min(clean_count / orig_count, 1.0) if orig_count > 0 else 1.0
                fidelity_score += ratio * weight
            return fidelity_score * 100

class ConversionPipeline:
    def __init__(self):
        self.quality_analyzer = QualityAnalyzer()
        self.markdown_processor = MarkdownPostProcessor()
        self.converters: Dict[str, ConverterEngine] = {
            "crawl4ai": Crawl4AIConverter(),
            "readability": ReadabilityConverter(),
            "jina": JinaConverter(),
            "html-to-markdown": HtmlToMarkdownConverter(),
        }
        logger.info("ConversionPipeline initialized.")
    @error_handler(logger)
    async def process(self, content: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        start_time = time.time()
        def _update_progress(progress: int, message: str):
            if progress_callback:
                progress_callback(progress, message)
        _update_progress(10, "Detecting input type...")
        input_type, normalized_content = InputTypeDetector.detect(content)
        converter_name = config_manager.get(ConfigKeys.PRIMARY_CONVERTER)
        converter = self.converters.get(converter_name)
        if not converter:
            raise ValueError(f"Unknown converter: {converter_name}")
        content_to_convert = ""
        if converter_name == "jina":
            if input_type != "url":
                raise ValueError("Jina engine only supports URL input")
            content_to_convert = normalized_content
            _update_progress(25, "Prepared Jina URL")
        elif converter_name == "html-to-markdown":
            if input_type == "url":
                _update_progress(15, "Downloading HTML content...")
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
                    }
                    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
                        response = await client.get(normalized_content, timeout=config_manager.get(ConfigKeys.TIMEOUT))
                        response.raise_for_status()
                        content_to_convert = response.text
                    _update_progress(25, "Downloaded content.")
                except Exception as e:
                    logger.error(f"URL download failed: {e}")
                    raise
            elif input_type == "file":
                path = Path(normalized_content)
                max_size = config_manager.get(ConfigKeys.MAX_FILE_SIZE_MB) * 1024 * 1024
                if path.stat().st_size > max_size:
                    raise ValueError(f"File too large: {max_size // 1024 // 1024}MB limit.")
                with path.open('r', encoding='utf-8', errors='ignore') as f:
                    content_to_convert = f.read()
                _update_progress(25, f"Loaded file: {path.name}")
            else:
                content_to_convert = normalized_content
                _update_progress(25, "Loaded raw HTML/text.")
        elif input_type == "url":
            if converter_name == "crawl4ai":
                _update_progress(20, "Crawl4AI handles URLs itself.")
                content_to_convert = normalized_content
            else:
                _update_progress(15, f"Downloading content for {converter_name}...")
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
                    }
                    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
                        response = await client.get(normalized_content, timeout=config_manager.get(ConfigKeys.TIMEOUT))
                        response.raise_for_status()
                        content_to_convert = response.text
                    _update_progress(25, "Downloaded content.")
                except Exception as e:
                    logger.error(f"URL download failed: {e}")
                    raise
        elif input_type == "file":
            path = Path(normalized_content)
            max_size = config_manager.get(ConfigKeys.MAX_FILE_SIZE_MB) * 1024 * 1024
            if path.stat().st_size > max_size:
                raise ValueError(f"File too large: {max_size // 1024 // 1024}MB limit.")
            with path.open('r', encoding='utf-8', errors='ignore') as f:
                content_to_convert = f.read()
            _update_progress(25, f"Loaded file: {path.name}")
        else:
            content_to_convert = normalized_content
            _update_progress(25, "Loaded raw HTML/text.")
        _update_progress(30, f"Converting with {converter_name}...")
        result = await converter.convert(content_to_convert)
        _update_progress(70, "Conversion done, analyzing quality...")
        if not result or not result.get('success'):
            return result or {'error': f'{converter_name} failed.', 'success': False}
        if result.get('markdown'):
            _update_progress(80, "Cleaning markdown...")
            original_markdown = result['markdown']
            cleaned_markdown = self.markdown_processor.clean_markdown(original_markdown)
            result['markdown'] = cleaned_markdown
            result['original_markdown'] = original_markdown
        _update_progress(90, "Calculating metrics...")
        original_html = result.get('html', content_to_convert if '<html' in content_to_convert else '')
        metrics = self.quality_analyzer.calculate(original_html, result, time.time() - start_time)
        result['metrics'] = metrics
        _update_progress(100, "Done.")
        return result

@dataclass
class ConversionTask:
    content: str
    name: str
    observers: List['ProgressObserver']
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    def notify(self, event: str, *args, **kwargs):
        for observer in self.observers:
            handler = getattr(observer, f"on_{event}", None)
            if handler:
                handler(*args, **kwargs)

class TaskManager:
    def __init__(self, pipeline: ConversionPipeline):
        self.pipeline = pipeline
        self.task_queue = queue.Queue()
        self.is_running = False
    def add_task(self, task: ConversionTask):
        self.task_queue.put(task)
        if not self.is_running:
            self.is_running = True
            threading.Thread(target=self._process_queue, daemon=True).start()
    def _process_queue(self):
        while not self.task_queue.empty():
            task = self.task_queue.get()
            try:
                task.status = "processing"
                task.notify("progress", 0, f"Start: {task.name}")
                result = asyncio.run(self.pipeline.process(
                    task.content,
                    lambda p, m: task.notify("progress", p, m)
                ))
                task.result = result
                task.status = "completed" if result.get('success') else "failed"
                if result.get('success'):
                    task.notify("complete", result)
                else:
                    task.notify("error", Exception(result.get('error', 'Unknown error')))
            except Exception as e:
                logger.error(f"Task {task.name} failed: {e}", exc_info=True)
                task.status = "failed"
                task.notify("error", e)
            self.task_queue.task_done()
        self.is_running = False

class ProgressObserver(ABC):
    @abstractmethod
    def on_progress(self, progress: int, message: str): pass
    @abstractmethod
    def on_complete(self, result: Dict[str, Any]): pass
    @abstractmethod
    def on_error(self, error: Exception): pass

class ProfessionalGUI(ProgressObserver):
    def __init__(self, root: tk.Tk):
        self.root = root
        self.pipeline = ConversionPipeline()
        self.task_manager = TaskManager(self.pipeline)
        self.current_task: Optional[ConversionTask] = None
        self._setup_ui()
        logger.info("GUI ready.")
    def _on_close(self):
        config_manager.save()
        self.root.destroy()
    def _setup_ui(self):
        self.root.title("HTML→Markdown Converter (Google Engineering Optimized)")
        self.root.geometry("1000x800")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.rowconfigure(2, weight=1)
        main_frame.columnconfigure(0, weight=1)
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(header_frame, text="HTML → Markdown Converter", font=("Arial", 16, "bold")).pack(side=tk.LEFT)
        ttk.Button(header_frame, text="清洗设置", command=self._open_cleaning_settings).pack(side=tk.RIGHT, padx=5)
        self.converter_var = tk.StringVar(value=config_manager.get(ConfigKeys.PRIMARY_CONVERTER))
        converter_combo = ttk.Combobox(header_frame, textvariable=self.converter_var, values=list(self.pipeline.converters.keys()), state="readonly", width=20)
        converter_combo.pack(side=tk.RIGHT, padx=5)
        converter_combo.bind("<<ComboboxSelected>>", self._on_converter_changed)
        ttk.Label(header_frame, text="主转换引擎:").pack(side=tk.RIGHT)
        input_frame = self._create_input_frame(main_frame)
        input_frame.grid(row=1, column=0, sticky="ew", pady=10)
        results_frame = self._create_results_frame(main_frame)
        results_frame.grid(row=2, column=0, sticky="nsew", pady=10)
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        status_frame.columnconfigure(0, weight=1)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var)
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var).grid(row=1, column=0, sticky="w")
    def _create_input_frame(self, parent) -> ttk.Frame:
        frame = ttk.LabelFrame(parent, text="输入", padding=15)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text="输入 URL、文件路径或直接粘贴 HTML 文件路径:").grid(row=0, column=0, columnspan=2, sticky="w")
        self.input_var = tk.StringVar()
        input_entry = ttk.Entry(frame, textvariable=self.input_var, font=("Consolas", 10))
        input_entry.grid(row=1, column=0, sticky="ew", pady=5)
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=1, column=1, padx=(10, 0))
        ttk.Button(button_frame, text="浏览文件...", command=self._select_file).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="开始转换", command=self._process_input).pack(side=tk.LEFT, padx=5)
        return frame
    def _create_results_frame(self, parent) -> ttk.Frame:
        frame = ttk.LabelFrame(parent, text="结果", padding=15)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=0, column=0, sticky="w", pady=(0, 10))
        self.result_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=("Consolas", 10), state=tk.DISABLED)
        self.result_text.grid(row=1, column=0, sticky="nsew")
        ttk.Button(button_frame, text="Markdown", command=lambda: self._show_content("markdown")).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="原始Markdown", command=lambda: self._show_content("original_markdown")).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="清理后HTML", command=lambda: self._show_content("cleaned_html")).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="分析报告", command=lambda: self._show_content("metrics")).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="保存 Markdown...", command=self._save_result).pack(side=tk.RIGHT)
        return frame
    def _on_converter_changed(self, event=None):
        config_manager.set(ConfigKeys.PRIMARY_CONVERTER, self.converter_var.get())
        self.on_progress(0, f"Converter switched to: {self.converter_var.get()}")
    def _select_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("HTML 文件", "*.html;*.htm"), ("所有文件", "*.*")])
        if filepath:
            self.input_var.set(filepath)
    def _process_input(self):
        content = self.input_var.get().strip()
        if not content:
            messagebox.showwarning("输入错误", "请输入有效的 URL、文件路径或 HTML 内容。")
            return
        task_name = f"处理: {content[:50]}" + ("..." if len(content) > 50 else "")
        self.current_task = ConversionTask(content=content, name=task_name, observers=[self])
        self.task_manager.add_task(self.current_task)
    def _save_result(self):
        if not (self.current_task and self.current_task.result):
            messagebox.showwarning("无法保存", "没有可供保存的结果。")
            return
        markdown = self.current_task.result.get("markdown")
        if not markdown:
            messagebox.showwarning("无法保存", "结果中不包含 Markdown 内容。")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".md", filetypes=[("Markdown 文件", "*.md"), ("文本文件", "*.txt")])
        if filepath:
            try:
                Path(filepath).write_text(markdown, encoding='utf-8')
                messagebox.showinfo("保存成功", f"结果已成功保存到:\n{filepath}")
            except Exception as e:
                messagebox.showerror("保存失败", f"无法写入文件: {e}")
    def _show_content(self, content_type: str):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete('1.0', tk.END)
        if not (self.current_task and self.current_task.result):
            self.result_text.insert('1.0', "尚无结果。")
            self.result_text.config(state=tk.DISABLED)
            return
        display_content = "内容不可用。"
        if content_type == "metrics":
            metrics = self.current_task.result.get('metrics')
            if isinstance(metrics, ConversionMetrics):
                original_md = self.current_task.result.get('original_markdown', '')
                cleaned_md = self.current_task.result.get('markdown', '')
                cleaning_enabled = config_manager.get(ConfigKeys.ENABLE_MARKDOWN_CLEANING, True)
                display_content = f"""--- 分析报告 ---
转换引擎: {config_manager.get(ConfigKeys.PRIMARY_CONVERTER)}
任务名称: {self.current_task.name}

--- 质量指标 ---
保真度 (Fidelity): {metrics.fidelity:.2f}%
广告残留率 (Ad Retention): {metrics.ad_retention:.2f}%
单词计数 (Word Count): {metrics.word_count}
总耗时 (Elapsed Time): {metrics.elapsed_time:.2f}秒

--- Markdown清洗信息 ---
清洗功能: {'启用' if cleaning_enabled else '禁用'}
原始长度: {len(original_md)} 字符
清洗后长度: {len(cleaned_md)} 字符
压缩率: {((len(original_md) - len(cleaned_md)) / max(len(original_md), 1) * 100):.1f}% (如果启用清洗)

--- 附加信息 ---
任务状态: {'成功' if self.current_task.result.get('success') else '失败'}
来源 URL: {self.current_task.result.get('url', 'N/A')}
HTTP 状态码: {self.current_task.result.get('status_code', 'N/A')}
"""
            else:
                display_content = "分析报告不可用。"
        else:
            if content_type == "original_markdown":
                content = self.current_task.result.get("original_markdown")
            else:
                content = self.current_task.result.get(content_type)
            if isinstance(content, (dict, list)):
                display_content = json.dumps(content, indent=2, ensure_ascii=False)
            elif content:
                display_content = str(content)
        self.result_text.insert('1.0', display_content)
        self.result_text.config(state=tk.DISABLED)
    def on_progress(self, progress: int, message: str):
        self.root.after(0, self._update_progress_ui, progress, message)
    def on_complete(self, result: Dict[str, Any]):
        self.root.after(0, self._update_ui_on_complete, result)
    def on_error(self, error: Exception):
        self.root.after(0, self._update_ui_on_error, error)
    def _update_progress_ui(self, progress: int, message: str):
        self.progress_var.set(progress)
        self.status_var.set(message)
    def _update_ui_on_complete(self, result: Dict[str, Any]):
        self.on_progress(100, "任务成功完成！")
        self._show_content("markdown")
    def _update_ui_on_error(self, error: Exception):
        self.on_progress(0, f"任务失败: {error}")
        messagebox.showerror("任务失败", str(error))
    def _open_cleaning_settings(self):
        settings_dialog = tk.Toplevel(self.root)
        settings_dialog.title("Markdown清洗设置")
        settings_dialog.geometry("700x700")
        settings_dialog.transient(self.root)
        settings_dialog.grab_set()
        main_frame = ttk.Frame(settings_dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        settings_frame = ttk.LabelFrame(main_frame, text="清洗设置", padding=15)
        settings_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        preview_frame = ttk.LabelFrame(main_frame, text="预览效果", padding=15)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(settings_frame)
        scrollbar = ttk.Scrollbar(settings_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        preview_text = scrolledtext.ScrolledText(preview_frame, height=15, wrap=tk.WORD, font=("Consolas", 9))
        preview_text.pack(fill=tk.BOTH, expand=True)
        def update_preview():
            if (self.current_task and self.current_task.result and 
                self.current_task.result.get('original_markdown')):
                original_md = self.current_task.result['original_markdown']
                try:
                    cleaned_md = self.pipeline.markdown_processor.clean_markdown(original_md)
                    preview_text.delete('1.0', tk.END)
                    preview_text.insert('1.0', f"=== 原始长度: {len(original_md)} 字符 ===\n")
                    preview_text.insert(tk.END, f"=== 清洗后长度: {len(cleaned_md)} 字符 ===\n\n")
                    preview_text.insert(tk.END, cleaned_md[:1000] + "..." if len(cleaned_md) > 1000 else cleaned_md)
                except Exception as e:
                    preview_text.delete('1.0', tk.END)
                    preview_text.insert('1.0', f"预览失败: {e}")
            else:
                preview_text.delete('1.0', tk.END)
                preview_text.insert('1.0', "暂无可预览的内容，请先执行一次转换。")
        enable_var = tk.BooleanVar(value=config_manager.get(ConfigKeys.ENABLE_MARKDOWN_CLEANING, True))
        def update_enable_setting():
            self._update_config(ConfigKeys.ENABLE_MARKDOWN_CLEANING, enable_var.get())
            update_preview()
        ttk.Checkbutton(scrollable_frame, text="启用Markdown清洗", variable=enable_var,
                        command=update_enable_setting)\
            .grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        options = [
            (ConfigKeys.MIN_LINE_LENGTH, "最小行长度（字符）", "spinbox", 10, 200, 5),
            (ConfigKeys.REMOVE_SHORT_LINES, "删除短行", "check"),
            (ConfigKeys.REMOVE_FOOTNOTES, "删除脚注", "check"),
            (ConfigKeys.REMOVE_DUPLICATE_HEADLINES, "删除重复标题", "check"),
            (ConfigKeys.CONTRACT_EMPTY_LINES, "压缩空行", "check"),
            (ConfigKeys.CRIMP_LINEBREAKS, "优化换行符", "check"),
        ]
        row = 1
        for key, label, widget_type, *args in options:
            ttk.Label(scrollable_frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=2)
            if widget_type == "spinbox":
                min_val, max_val, increment = args
                var = tk.IntVar(value=config_manager.get(key, min_val))
                def update_spinbox_setting(k=key, v=var):
                    self._update_config(k, v.get())
                    update_preview()
                spinbox = ttk.Spinbox(scrollable_frame, from_=min_val, to=max_val, increment=increment, 
                                    textvariable=var, width=10)
                spinbox.grid(row=row, column=1, sticky="e", pady=2)
                spinbox.bind("<FocusOut>", lambda e, func=update_spinbox_setting: func())
            elif widget_type == "check":
                var = tk.BooleanVar(value=config_manager.get(key, True))
                def update_check_setting(k=key, v=var):
                    self._update_config(k, v.get())
                    update_preview()
                check = ttk.Checkbutton(scrollable_frame, variable=var, command=update_check_setting)
                check.grid(row=row, column=1, sticky="e", pady=2)
            row += 1
        update_preview()
        ttk.Button(settings_frame, text="关闭", 
                command=settings_dialog.destroy).pack(side=tk.BOTTOM, pady=(10, 0))
    def _update_config(self, key: ConfigKeys, value: Any):
        config_manager.set(key, value)
        if hasattr(self, 'pipeline') and hasattr(self.pipeline, 'markdown_processor'):
            self.pipeline.markdown_processor.refresh_cleaner()
        if (self.current_task and self.current_task.result and 
            self.current_task.result.get('original_markdown')):
            try:
                original_md = self.current_task.result['original_markdown']
                cleaned_md = self.pipeline.markdown_processor.clean_markdown(original_md)
                self.current_task.result['markdown'] = cleaned_md
                original_html = self.current_task.result.get('html', '')
                start_time = time.time()
                metrics = self.pipeline.quality_analyzer.calculate(
                    original_html, self.current_task.result, time.time() - start_time)
                self.current_task.result['metrics'] = metrics
                self._show_content("markdown")
                self.on_progress(100, f"清洗设置已更新并应用: {key.value} = {value}")
            except Exception as e:
                logger.error(f"Apply cleaning failed: {e}")
                self.on_progress(0, f"清洗设置已更新: {key.value} = {value}")
        else:
            self.on_progress(0, f"清洗设置已更新: {key.value} = {value}")

def main():
    global config_manager
    config_manager = ConfigManager()
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        if sys.platform == "win32":
            style.theme_use('vista')
        elif sys.platform == "darwin":
            style.theme_use('aqua')
        else:
            style.theme_use('clam')
    except tk.TclError:
        logger.warning("Unable to set ttk theme, using default.")
    style.configure("Accent.TButton", font=('Arial', 10, 'bold'), foreground="white", background="#0078D4")
    app = ProfessionalGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()