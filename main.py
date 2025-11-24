import sys
import json
import re
import asyncio
import threading
import csv
from datetime import datetime
from typing import List
import traceback
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox, QProgressBar, QLineEdit,
    QSplitter, QHeaderView, QStatusBar, QGroupBox, QTabWidget
)
from PySide6.QtCore import (
    Qt, QThread, Signal, QObject, QTimer
)
from PySide6.QtGui import (QColor, QPalette, QIcon, QPixmap)
from PySide6.QtWidgets import QSizePolicy
from crawl4ai import AsyncWebCrawler, CacheMode
from openai import OpenAI, APIError
import os
import PyPDF2
import pdfplumber
PDF_SUPPORT = True

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class CrawlWorker(QObject):
    """Asynchronous crawling worker thread"""
    finished = Signal(dict)
    progress = Signal(str, int)
    error = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.crawler = None
        self.is_running = False
    
    async def run_crawl(self, urls: List[str], config: dict):
        """Execute crawling task"""
        
        try:
            self.is_running = True
            results = []
            
            # Map index to CacheMode enum
            cache_mode_map = {
                0: CacheMode.DISABLED,  # No cache
                1: CacheMode.ENABLED,   # Memory cache
                2: CacheMode.ENABLED    # Disk cache (crawl4ai's ENABLED usually includes memory and disk)
            }
            cache_mode_index = config.get('cache_mode', 1)
            cache_mode = cache_mode_map.get(cache_mode_index, CacheMode.ENABLED)
            
            async with AsyncWebCrawler(
                cache_mode=cache_mode,
                headless=not config.get('headful', False),
                verbose=config.get('verbose', True)
            ) as crawler:
                self.crawler = crawler
                
                total_urls = len(urls)
                for i, url in enumerate(urls, 1):
                    if not self.is_running:
                        break
                    
                    self.progress.emit(f"Crawling: {url}", int(i / total_urls * 100))
                    
                    try:
                        result = await crawler.arun(
                            url=url,
                            timeout=config.get('timeout', 30),
                            max_depth=config.get('max_depth', 1),
                            exclude_urls=config.get('exclude_urls', []),
                            extract_markdown=config.get('extract_markdown', True),
                            extract_html=config.get('extract_html', False),
                            generate_metadata=config.get('generate_metadata', True),
                            screenshot=config.get('screenshot', False),
                            magic=config.get('magic', False),  # AI enhanced extraction
                            **config.get('extra_args', {})
                        )
                        
                        if result.success:
                            results.append({
                                'url': url,
                                'title': result.metadata.get('title', 'Untitled'),
                                'markdown': result.markdown if hasattr(result, 'markdown') else '',
                                'html': result.html if hasattr(result, 'html') else '',
                                'metadata': result.metadata,
                                'success': True,
                                'timestamp': datetime.now().isoformat()
                            })
                        else:
                            results.append({
                                'url': url,
                                'error': 'Crawling failed',
                                'success': False
                            })
                            
                    except Exception as e:
                        results.append({
                            'url': url,
                            'error': str(e),
                            'success': False
                        })
            
            self.finished.emit({'results': results, 'config': config})
            
        except Exception as e:
            self.error.emit(f"Error during crawling: {str(e)}")
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop crawling"""
        self.is_running = False

class PDFWorker(QObject):
    """PDF processing worker thread"""
    finished = Signal(dict)
    progress = Signal(str, int)
    error = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.is_running = False
    
    def process_pdfs(self, pdf_paths: List[str], config: dict):
        """Process PDF files and extract text content"""
        if not PDF_SUPPORT:
            self.error.emit("PDF support libraries not available. Please install PyPDF2 and pdfplumber.")
            return
        
        self.is_running = True
        results = []
        
        try:
            total_files = len(pdf_paths)
            for i, pdf_path in enumerate(pdf_paths, 1):
                if not self.is_running:
                    break
                
                file_name = os.path.basename(pdf_path)
                self.progress.emit(f"Processing PDF: {file_name}", int(i / total_files * 100))
                
                try:
                    # Try using pdfplumber first (better text extraction)
                    text_content = ""
                    metadata = {}
                    
                    try:
                        with pdfplumber.open(pdf_path) as pdf:
                            metadata = pdf.metadata or {}
                            for page in pdf.pages:
                                if not self.is_running:
                                    break
                                page_text = page.extract_text()
                                if page_text:
                                    text_content += page_text + "\n\n"
                    except Exception as e:
                        print(f"pdfplumber failed for {pdf_path}: {str(e)}, trying PyPDF2...")
                        
                        # Fallback to PyPDF2
                        try:
                            with open(pdf_path, 'rb') as file:
                                pdf_reader = PyPDF2.PdfReader(file)
                                metadata = pdf_reader.metadata or {}
                                
                                for page_num in range(len(pdf_reader.pages)):
                                    if not self.is_running:
                                        break
                                    page = pdf_reader.pages[page_num]
                                    text_content += page.extract_text() + "\n\n"
                        except Exception as e2:
                            raise Exception(f"Both pdfplumber and PyPDF2 failed: {str(e2)}")
                    
                    if text_content.strip():
                        results.append({
                            'url': f"file://{pdf_path}",
                            'title': metadata.get('Title', file_name),
                            'markdown': text_content,
                            'html': '',
                            'metadata': {
                                'source': 'pdf',
                                'file_path': pdf_path,
                                'file_name': file_name,
                                **{k: v for k, v in metadata.items() if isinstance(v, str)}
                            },
                            'success': True,
                            'timestamp': datetime.now().isoformat()
                        })
                    else:
                        results.append({
                            'url': f"file://{pdf_path}",
                            'error': 'No text content extracted',
                            'success': False
                        })
                        
                except Exception as e:
                    results.append({
                        'url': f"file://{pdf_path}",
                        'error': str(e),
                        'success': False
                    })
            
            self.finished.emit({'results': results, 'config': config})
            
        except Exception as e:
            self.error.emit(f"Error during PDF processing: {str(e)}")
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop PDF processing"""
        self.is_running = False

class LLMWorker(QObject):
    """LLM processing worker thread"""
    finished = Signal(dict)
    progress = Signal(str, int)
    error = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.is_running = False
        self.client = None
    
    def init_client(self, provider, api_key, base_url=None):
        """Initialize LLM client (unified through OpenAI compatible interface)"""
        return OpenAI(
            api_key=api_key,
            base_url=base_url.strip()
        )
    
    async def call_llm(self, provider, api_key, messages, base_url=None, extra_params=None):
        """Call LLM API (unified through OpenAI compatible interface)"""
        try:
            self.client = self.init_client(provider, api_key, base_url)
            # Parse extra parameters
            extra_body = {}
            if extra_params:
                try:
                    extra_body = json.loads(extra_params)
                except json.JSONDecodeError as e:
                    print(f"Error parsing extra parameters: {e}")
            # Determine model name
            model_name = provider.split('/')[-1] if '/' in provider else provider
            # Call API
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,
                extra_body=extra_body
            )
            return completion.choices[0].message.content
        except APIError as e:
            error_msg = str(e)
            # Remove error messages that might contain API key
            if "api_key" in error_msg.lower():
                error_msg = "API call failed, please check API key and network connection"
            raise Exception(f"LLM API error: {error_msg}")
        except Exception as e:
            raise Exception(f"LLM API call failed: {str(e)}")
    
    def process_contents(self, contents, provider, api_key, base_url, system_prompt, prompt_template, extra_params):
        """Process content to generate training data"""
        # Threading/asyncio has been moved to the top of the file
        
        self.is_running = True
        
        async def process_all():
            results = []
            total = len(contents)
            
            for i, item in enumerate(contents):
                if not self.is_running:
                    break
                
                self.progress.emit(f"Processing: {item['title'] or item['url']}", int(i / total * 100))
                
                try:
                    # Prepare prompt
                    content = item['content']
                    prompt = prompt_template.replace("{content}", content)
                    
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    
                    # Call LLM
                    response = await self.call_llm(
                        provider=provider,
                        api_key=api_key,
                        messages=messages,
                        base_url=base_url,
                        extra_params=extra_params
                    )
                    
                    # Extract JSON data (handle possible Markdown format)
                    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        json_str = response
                    
                    # Parse JSON
                    try:
                        data = json.loads(json_str)
                        if isinstance(data, list):
                            # Add metadata
                            for sample in data:
                                # Ensure required fields exist
                                if "instruction" not in sample:
                                    sample["instruction"] = "Answer questions based on the content"
                                if "output" not in sample:
                                    sample["output"] = "No answer"
                            results.extend(data)
                        else:
                            print(f"Return from URL {i+1} is not in list format, skipping")
                    except json.JSONDecodeError as e:
                        print(f"JSON parsing error: {e}")
                        print(f"Response content: {json_str}")
                        # Try to fix JSON
                        fixed_data = self.fix_json_response(json_str)
                        if fixed_data:
                            results.extend(fixed_data)
                        else:
                            print(f"Unable to fix JSON data from URL {i+1}, skipping")
                    
                except Exception as e:
                    print(f"Error processing {item['url']}: {str(e)}")
                    continue
            
            return {'data': results}
        
        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(process_all())
                self.finished.emit(result)
            except Exception as e:
                self.error.emit(f"LLM processing error: {str(e)}")
            finally:
                loop.close()
                self.is_running = False
        
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
    
    def fix_json_response(self, response):
        """Try to fix invalid JSON response"""
        try:
            # Remove comments
            response = re.sub(r'//.*?\n|/\*.*?\*/', '', response, flags=re.DOTALL)
            # Try to parse
            return json.loads(response)
        except:
            try:
                # Try to extract the first valid JSON object
                json_str = re.search(r'\[[\s\S]*\]|\{[\s\S]*\}', response)
                if json_str:
                    return json.loads(json_str.group(0))
            except:
                return None
        return None
    
    def stop(self):
        """Stop processing"""
        self.is_running = False

class ModernCrawlerUI(QMainWindow):
    """Modern crawler interface"""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(1200, 800)
        
        # Setup styles
        self.setup_styles()
        
        # Initialize data (needs to be initialized before setup_ui as setup_ui calls update_llm_buttons)
        self.results = []
        self.current_result = None
        self.training_data = []
        self.pdf_files = []  # Store selected PDF file paths
        
        # Fixed prompt template
        self.prompt_template = """请基于以下网页内容生成训练数据。要求：
1. 分析网页的主要内容和结构
2. 提取有意义的问答对或指令-响应对
3. 每个样本应包含清晰的指令和高质量的回答
4. 如果内容适合多轮对话，可以生成带有history字段的样本
5. 确保数据对LLM训练有价值
6. 仅整理通识性信息，不要整理专有信息（如"这篇文章...""本研究......"之类的问题）
7. 确保生成的训练数据内容为中文
8. 基于提供的完整内容生成训练数据，不要遗漏任何重要信息

请严格按照以下JSON格式输出：
[
  {
    "instruction": "用户指令（必填）",
    "input": "用户输入（可选）",
    "output": "模型回答（必填）",
    "system": "系统提示（可选）",
    "history": [
      ["第一轮指令（可选）", "第一轮回答（可选）"],
      ["第二轮指令（可选）", "第二轮回答（可选）"]
    ]
  }
]
**请注意每个数据项必须至少包含"instruction"、"input"和"output"三个字段。不要添加除上述字段外的任何字段**
网页内容：
{content}"""
        
        # Create UI
        self.setup_ui()
        
        # Create crawling worker thread
        self.thread = QThread()
        self.worker = CrawlWorker()
        self.worker.moveToThread(self.thread)
        
        # Connect signals
        self.worker.finished.connect(self.on_crawl_finished)
        self.worker.progress.connect(self.update_progress)
        self.worker.error.connect(self.show_error)
        
        self.thread.start()
        
        # Create LLM worker thread
        self.llm_thread = QThread()
        self.llm_worker = LLMWorker()
        self.llm_worker.moveToThread(self.llm_thread)
        
        # Connect LLM signals
        self.llm_worker.finished.connect(self.on_training_data_generated)
        self.llm_worker.progress.connect(self.update_progress)
        self.llm_worker.error.connect(self.show_error)
        
        self.llm_thread.start()
        
        # Create PDF worker thread
        self.pdf_thread = QThread()
        self.pdf_worker = PDFWorker()
        self.pdf_worker.moveToThread(self.pdf_thread)
        
        # Connect PDF signals
        self.pdf_worker.finished.connect(self.on_pdf_processing_finished)
        self.pdf_worker.progress.connect(self.update_progress)
        self.pdf_worker.error.connect(self.show_error)
        
        self.pdf_thread.start()
    
    def setup_styles(self):
        """Setup interface styles"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QWidget {
                font-family: 'Segoe UI', 'Microsoft YaHei', Arial, sans-serif;
                font-size: 14px;
            }
            QLineEdit, QTextEdit, QTableWidget {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 6px;
                background-color: white;
            }
            QPushButton {
                background-color: #4a6fa5;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a5a80;
            }
            QPushButton:disabled {
                background-color: #a0b8d0;
            }
            QPushButton#primary {
                background-color: #2196f3;
            }
            QPushButton#primary:hover {
                background-color: #0d8aee;
            }
            QPushButton#success {
                background-color: #4caf50;
            }
            QPushButton#danger {
                background-color: #f44336;
            }
            QTabWidget::pane {
                border: 1px solid #ddd;
                background: white;
                border-radius: 4px;
            }
            QTabBar::tab {
                background: #e0e0e0;
                border: 1px solid #ccc;
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom-color: white;
            }
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2196f3;
                width: 10px;
            }
            QTableWidget {
                gridline-color: #eee;
                selection-background-color: #2196f3;
                selection-color: white;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 6px;
                border: 1px solid #ddd;
                font-weight: bold;
            }
            QLabel#status {
                color: #666;
                font-style: italic;
            }
            QFrame#line {
                background-color: #ddd;
                height: 1px;
                margin: 5px 0;
            }
            QGroupBox {
                border: 1px solid #ddd;
                border-radius: 8px;
                margin-top: 1ex;
                font-weight: bold;
                color: #2c3e50;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
    
    def setup_ui(self):
        """Setup user interface"""
        # Central widget
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Title row (horizontal layout)
        title_row_layout = QHBoxLayout()
        
        # Create icon label
        icon_label = QLabel()
        icon_pixmap = QPixmap(resource_path("sign.png"))
        icon_label.setPixmap(icon_pixmap.scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        
        # Title
        title_label = QLabel("CapraData")
        title_label.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
            padding-top: 15px;
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        # Add icon and title to horizontal layout
        title_row_layout.addWidget(icon_label)
        title_row_layout.addWidget(title_label)
        
        # Add stretch
        title_row_layout.addStretch()
        
        # Model input
        model_layout = QVBoxLayout()
        model_label = QLabel("Model:")
        model_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("Enter model name")
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_input)
        title_row_layout.addLayout(model_layout)
        
        # API Key input
        api_key_layout = QVBoxLayout()
        api_key_label = QLabel("API Key:")
        api_key_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Enter API key")
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        api_key_layout.addWidget(api_key_label)
        api_key_layout.addWidget(self.api_key_input)
        title_row_layout.addLayout(api_key_layout)
        
        main_layout.addLayout(title_row_layout)
        
        # Create main splitter (horizontal split)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: crawl configuration and results
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        
        # Create tab widget for different data sources
        self.source_tabs = QTabWidget()
        
        # Web crawling tab
        crawl_tab = QWidget()
        crawl_layout = QVBoxLayout(crawl_tab)
        
        # Crawl configuration area
        self.setup_crawl_config(crawl_layout)
        
        # PDF processing tab
        pdf_tab = QWidget()
        pdf_layout = QVBoxLayout(pdf_tab)
        
        # PDF processing configuration area
        self.setup_pdf_config(pdf_layout)
        
        # Add tabs to the tab widget
        self.source_tabs.addTab(crawl_tab, "🌐 Web Crawling")
        self.source_tabs.addTab(pdf_tab, "📄 PDF Processing")
        
        left_layout.addWidget(self.source_tabs)
        
        # Results area (from results page)
        self.setup_results_section(left_layout)
        
        # Right panel: LLM data generation
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        
        # LLM data generation area
        self.setup_llm_section(right_layout)
        
        # Add panels to splitter
        # Set both panels to expand as much as possible and distribute proportionally
        left_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 3)
        
        main_layout.addWidget(main_splitter)
        # Give more space to the splitter area
        if isinstance(main_layout, QVBoxLayout):
            main_layout.setStretch(0, 0)  # Title
            main_layout.setStretch(1, 1)  # Splitter
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(25)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar, 1)
        
        self.setCentralWidget(central_widget)
        
        # Update LLM button status
        self.update_llm_buttons()
    
    def setup_pdf_config(self, parent_layout):
        """Setup PDF processing configuration area"""
        # PDF file selection area
        pdf_group = QGroupBox("📄 PDF File Selection")
        pdf_layout = QVBoxLayout(pdf_group)
        pdf_layout.setContentsMargins(10, 10, 10, 10)
        pdf_layout.setSpacing(8)
        
        # File selection button
        file_select_layout = QHBoxLayout()
        self.select_pdf_btn = QPushButton("Select PDF Files")
        self.select_pdf_btn.clicked.connect(self.select_pdf_files)
        file_select_layout.addWidget(self.select_pdf_btn)
        file_select_layout.addStretch()
        
        self.clear_pdf_btn = QPushButton("Clear PDF List")
        self.clear_pdf_btn.clicked.connect(self.clear_pdf_list)
        self.clear_pdf_btn.setEnabled(False)
        file_select_layout.addWidget(self.clear_pdf_btn)
        
        pdf_layout.addLayout(file_select_layout)
        
        # PDF file list display
        self.pdf_list_label = QLabel("No PDF files selected")
        self.pdf_list_label.setWordWrap(True)
        self.pdf_list_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        pdf_layout.addWidget(self.pdf_list_label)
        
        parent_layout.addWidget(pdf_group)
        
        # PDF processing options
        options_group = QGroupBox("⚙️ PDF Processing Options")
        options_layout = QVBoxLayout(options_group)
        options_layout.setContentsMargins(10, 10, 10, 10)
        
        # Add info label about PDF processing
        info_label = QLabel("PDF files will be processed to extract text content, which can then be used for training data generation.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #555; padding: 5px;")
        options_layout.addWidget(info_label)
        
        # Check if PDF support is available
        if not PDF_SUPPORT:
            warning_label = QLabel("⚠️ PDF support libraries not detected. Please install PyPDF2 and pdfplumber to enable PDF processing.")
            warning_label.setWordWrap(True)
            warning_label.setStyleSheet("color: #d9534f; padding: 5px; font-weight: bold;")
            options_layout.addWidget(warning_label)
            self.select_pdf_btn.setEnabled(False)
        
        parent_layout.addWidget(options_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.process_pdf_btn = QPushButton("Process PDF Files")
        self.process_pdf_btn.setObjectName("primary")
        self.process_pdf_btn.clicked.connect(self.start_pdf_processing)
        self.process_pdf_btn.setFixedHeight(40)
        self.process_pdf_btn.setEnabled(False)
        self.process_pdf_btn.setStyleSheet("font-size: 14px; padding: 8px 20px;")
        button_layout.addWidget(self.process_pdf_btn)
        
        self.clear_pdf_results_btn = QPushButton("Clear PDF Results")
        self.clear_pdf_results_btn.setObjectName("danger")
        self.clear_pdf_results_btn.clicked.connect(self.clear_pdf_results)
        self.clear_pdf_results_btn.setFixedHeight(40)
        self.clear_pdf_results_btn.setEnabled(False)
        self.clear_pdf_results_btn.setStyleSheet("font-size: 14px; padding: 8px 20px;")
        button_layout.addWidget(self.clear_pdf_results_btn)
        
        button_layout.addStretch()
        parent_layout.addLayout(button_layout)
    
    def setup_crawl_config(self, parent_layout):
        """Setup crawl configuration area"""
        # URL input area
        url_group = QGroupBox("🔗 Target URL Configuration")
        url_layout = QVBoxLayout(url_group)
        url_layout.setContentsMargins(10, 10, 10, 10)
        url_layout.setSpacing(8)
        
        url_label = QLabel("Enter URLs (one per line):")
        url_label.setFixedHeight(20)
        url_layout.addWidget(url_label)
        
        self.url_input = QTextEdit()
        # Set paste behavior to plain text
        self.url_input.setAcceptRichText(False)
        url_layout.addWidget(self.url_input)
        
        parent_layout.addWidget(url_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.start_btn = QPushButton("Generate Training Data")
        self.start_btn.setObjectName("primary")
        self.start_btn.setIcon(QIcon())
        self.start_btn.clicked.connect(self.start_generate_pipeline)
        self.start_btn.setFixedHeight(40)
        self.start_btn.setStyleSheet("font-size: 14px; padding: 8px 20px;")
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Crawling")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setFixedHeight(40)
        self.stop_btn.clicked.connect(self.stop_crawling)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("font-size: 14px; padding: 8px 20px;")
        button_layout.addWidget(self.stop_btn)

        self.clear_btn = QPushButton("Clear Results")
        self.clear_btn.setObjectName("danger")
        self.clear_btn.clicked.connect(self.clear_results)
        self.clear_btn.setEnabled(False)
        self.clear_btn.setFixedHeight(40)
        self.clear_btn.setStyleSheet("font-size: 14px; padding: 8px 20px;")
        button_layout.addWidget(self.clear_btn)
        
        # Add export training data button
        self.export_llm_btn = QPushButton("Export Training Data")
        self.export_llm_btn.setObjectName("success")
        self.export_llm_btn.clicked.connect(self.export_training_data)
        self.export_llm_btn.setEnabled(False)
        self.export_llm_btn.setFixedHeight(40)
        self.export_llm_btn.setStyleSheet("font-size: 14px; padding: 8px 20px;")
        button_layout.addWidget(self.export_llm_btn)
        
        button_layout.addStretch()
        parent_layout.addLayout(button_layout)
    
    def setup_results_section(self, parent_layout):
        """Setup results area"""
        # Results control bar
        control_layout = QHBoxLayout()
        
        control_layout.addStretch()
        parent_layout.addLayout(control_layout)
        
        # Results table (directly filled, removed splitter to avoid blank space)
        self.results_table = QTableWidget()
        self.results_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Status", "URL", "Title", "Time"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.results_table.itemSelectionChanged.connect(self.on_result_selected)
        parent_layout.addWidget(self.results_table)
    
    def setup_llm_section(self, parent_layout):
        """Setup LLM data generation area (model configuration removed, using default configuration)"""
        
        # Results preview
        preview_group = QGroupBox("🔍 Generated Training Data Preview")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(10, 10, 10, 10)
        
        self.training_data_preview = QTextEdit()
        self.training_data_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.training_data_preview.setReadOnly(True)
        self.training_data_preview.setStyleSheet("background-color: #f8f9fa;")
        preview_layout.addWidget(self.training_data_preview)
        
        parent_layout.addWidget(preview_group)
    
    def on_provider_changed(self, index):
        """Compatibility with old interface (configuration UI removed, no longer used)"""
        pass
    
    def select_pdf_files(self):
        """Open file dialog to select PDF files"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select PDF Files",
            "",
            "PDF Files (*.pdf);;All Files (*)"
        )
        
        if file_paths:
            self.pdf_files = file_paths
            self.update_pdf_list_display()
            self.process_pdf_btn.setEnabled(True)
            self.clear_pdf_btn.setEnabled(True)
    
    def clear_pdf_list(self):
        """Clear the selected PDF files list"""
        self.pdf_files = []
        self.update_pdf_list_display()
        self.process_pdf_btn.setEnabled(False)
        self.clear_pdf_btn.setEnabled(False)
    
    def update_pdf_list_display(self):
        """Update the PDF files list display"""
        if not self.pdf_files:
            self.pdf_list_label.setText("No PDF files selected")
        else:
            file_names = [os.path.basename(path) for path in self.pdf_files]
            if len(file_names) <= 5:
                self.pdf_list_label.setText(f"Selected {len(self.pdf_files)} PDF files:\n" + "\n".join(file_names))
            else:
                self.pdf_list_label.setText(f"Selected {len(self.pdf_files)} PDF files:\n" + "\n".join(file_names[:5]) + f"\n... and {len(self.pdf_files) - 5} more")
    
    def start_pdf_processing(self):
        """Start processing PDF files"""
        if not self.pdf_files:
            QMessageBox.warning(self, "No Files Selected", "Please select PDF files to process")
            return
        
        # Check if model and API key are provided
        provider = self.model_input.text().strip()
        api_key = self.api_key_input.text().strip()
        
        if not provider:
            QMessageBox.warning(self, "Input Error", "Please enter model name before processing PDF files")
            return
            
        if not api_key:
            QMessageBox.warning(self, "Input Error", "Please enter API key before processing PDF files")
            return
        
        # Set flag to automatically generate training data after PDF processing
        self.auto_generate_after_crawl = True
        
        # Update UI status
        self.process_pdf_btn.setEnabled(False)
        self.select_pdf_btn.setEnabled(False)
        self.status_label.setText("Processing PDF files...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        # Configuration parameters
        config = {
            'extract_markdown': True,
            'generate_metadata': True
        }
        
        # Start processing
        try:
            self.pdf_worker.process_pdfs(
                pdf_paths=self.pdf_files,
                config=config
            )
        except Exception as e:
            error_msg = f"Error starting PDF processing: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            QMessageBox.critical(self, "Error", error_msg)
            self.process_pdf_btn.setEnabled(True)
            self.select_pdf_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.status_label.setText("PDF processing startup failed")
    
    def on_pdf_processing_finished(self, result):
        """PDF processing completed"""
        pdf_results = result.get('results', [])
        self.results.extend(pdf_results)  # Add PDF results to existing results
        
        # Update results table
        self.update_results_table()
        
        # Update UI
        self.status_label.setText(f"Successfully processed {len([r for r in pdf_results if r.get('success')])} of {len(pdf_results)} PDF files")
        self.progress_bar.setValue(100)
        self.process_pdf_btn.setEnabled(True)
        self.select_pdf_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Enable clear PDF results button if there are PDF results
        self.clear_pdf_results_btn.setEnabled(len(pdf_results) > 0)
        
        # Update button states
        self.update_llm_buttons()
        
        # Show completion message - removed per user request
        success_count = len([r for r in pdf_results if r.get('success')])
        if success_count > 0:
            # If one-click process, automatically generate training data
            if getattr(self, 'auto_generate_after_crawl', False):
                self.auto_generate_after_crawl = False
                # Clear selection to ensure all PDF results are processed
                self.results_table.clearSelection()
                
                # Check if there are multiple PDF files
                if success_count > 1:
                    # Get indices of PDF results
                    pdf_indices = []
                    for i, r in enumerate(self.results):
                        if r.get('success') and r.get('url', '').startswith('file://') and r.get('url', '').endswith('.pdf'):
                            pdf_indices.append(i)
                    
                    # Process each PDF separately
                    self.process_each_pdf_separately(pdf_indices)
                else:
                    # Process as before for single PDF
                    self.generate_training_data()
        else:
            QMessageBox.warning(self, "Processing Issues", 
                               "No text content could be extracted from the selected PDF files.\n"
                               "The files might be scanned images or password-protected.")
    
    def process_each_pdf_separately(self, pdf_indices):
        """Process each PDF separately with individual LLM calls"""
        try:
            # Configure LLM (default + environment variables)
            provider, api_key, base_url, system_prompt, prompt_template, extra_params = self.get_default_llm_config()
            if not api_key:
                QMessageBox.warning(self, "API Key Missing", "No API key detected in environment variables, please set DASHSCOPE_API_KEY")
                return
            
            # Update UI status
            self.export_llm_btn.setEnabled(False)
            self.status_label.setText(f"Generating training data for {len(pdf_indices)} PDF files separately...")
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            
            # Store the indices for later use in callbacks
            self.pdf_indices = pdf_indices
            self.all_pdf_training_data = []
            self.current_pdf_index = 0
            
            # Start processing the first PDF
            self.process_next_pdf()
            
        except Exception as e:
            error_msg = f"Error in process_each_pdf_separately: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            QMessageBox.critical(self, "Error", error_msg)
            self.export_llm_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.status_label.setText("Processing failed")
    
    def process_next_pdf(self):
        """Process the next PDF in the list"""
        try:
            if self.current_pdf_index >= len(self.pdf_indices):
                # All PDFs have been processed
                self.on_all_pdfs_processed()
                return
            
            # Get the current PDF to process
            idx = self.pdf_indices[self.current_pdf_index]
            result = self.results[idx]
            
            if not result.get('success') or not result.get('markdown'):
                # Skip invalid results and move to the next
                self.current_pdf_index += 1
                self.process_next_pdf()
                return
            
            # Update progress
            progress = int((self.current_pdf_index / len(self.pdf_indices)) * 80)  # Use 80% for processing
            self.progress_bar.setValue(progress)
            self.status_label.setText(f"Processing PDF {self.current_pdf_index + 1}/{len(self.pdf_indices)}: {result.get('title', 'Untitled')}")
            
            # Configure LLM
            provider, api_key, base_url, system_prompt, prompt_template, extra_params = self.get_default_llm_config()
            
            # Prepare content for the current PDF
            content_item = {
                'url': result.get('url', ''),
                'title': result.get('title', ''),
                'content': result.get('markdown', '')
            }
            
            # Disconnect previous worker signals if they exist
            if hasattr(self, 'llm_worker'):
                try:
                    self.llm_worker.finished.disconnect()
                    self.llm_worker.error.disconnect()
                except:
                    pass
            
            # Create a new LLM worker for this PDF
            self.llm_worker = LLMWorker()
            self.llm_worker.finished.connect(self.on_single_pdf_processed)
            self.llm_worker.error.connect(self.on_single_pdf_error)
            
            # Start processing this PDF
            self.llm_worker.process_contents(
                contents=[content_item],  # Only one content item per call
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                system_prompt=system_prompt,
                prompt_template=prompt_template,
                extra_params=extra_params
            )
            
        except Exception as e:
            error_msg = f"Error in process_next_pdf: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            QMessageBox.critical(self, "Error", error_msg)
            self.export_llm_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.status_label.setText("Processing failed")
    
    def on_single_pdf_processed(self, result):
        """Callback for when a single PDF processing is completed"""
        try:
            # Add the result to our collection
            pdf_data = result.get('data', [])
            self.all_pdf_training_data.extend(pdf_data)
            
            # Disconnect the signals from the current worker
            try:
                self.llm_worker.finished.disconnect()
                self.llm_worker.error.disconnect()
            except:
                pass
            
            # Move to the next PDF
            self.current_pdf_index += 1
            
            # Process the next PDF
            self.process_next_pdf()
            
        except Exception as e:
            error_msg = f"Error processing single PDF result: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            QMessageBox.critical(self, "Error", error_msg)
            self.export_llm_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.status_label.setText("Processing failed")
    
    def on_all_pdfs_processed(self):
        """Callback for when all PDFs have been processed"""
        try:
            # Update progress to 100%
            self.progress_bar.setValue(100)
            self.status_label.setText(f"Successfully generated training data from {len(self.pdf_indices)} PDF files")
            
            # Set the training data
            self.training_data = self.all_pdf_training_data
            
            # Update UI
            self.export_llm_btn.setEnabled(len(self.training_data) > 0)
            self.progress_bar.setVisible(False)
            
            # Preview data
            preview_data = self.training_data
            preview_text = json.dumps(preview_data, indent=2, ensure_ascii=False)
            self.training_data_preview.setPlainText(preview_text)
            
            # Show completion message
            QMessageBox.information(self, "Processing Complete", 
                                  f"Successfully generated {len(self.training_data)} training data entries from {len(self.pdf_indices)} PDF files")
            
        except Exception as e:
            error_msg = f"Error in on_all_pdfs_processed: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            QMessageBox.critical(self, "Error", error_msg)
            self.export_llm_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.status_label.setText("Processing failed")
    
    def on_single_pdf_error(self, error_msg):
        """Callback for when a single PDF processing fails"""
        print(f"Error processing PDF: {error_msg}")
        
        # Disconnect the signals from the current worker
        try:
            self.llm_worker.finished.disconnect()
            self.llm_worker.error.disconnect()
        except:
            pass
        
        # Move to the next PDF even if the current one failed
        self.current_pdf_index += 1
        self.process_next_pdf()

    def clear_pdf_results(self):
        """Clear PDF processing results"""
        reply = QMessageBox.question(self, "Confirm Clear",
                                    "Are you sure you want to clear all PDF processing results?",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            # Filter out PDF results from the main results list
            self.results = [r for r in self.results if not r.get('url', '').endswith('.pdf')]
            
            # Update results table
            self.update_results_table()
            
            # Clear training data if it was generated from PDF results
            self.training_data = []
            self.training_data_preview.clear()
            
            # Update button states
            self.clear_pdf_results_btn.setEnabled(False)
            self.export_llm_btn.setEnabled(False)
            self.status_label.setText("PDF results cleared")
            self.update_llm_buttons()
    
    def update_llm_buttons(self):
        """Update LLM related button states"""
        has_results = len(self.results) > 0
        has_valid_results = any(r.get('success') and r.get('markdown') for r in self.results)
        
        self.export_llm_btn.setEnabled(has_valid_results)
        self.clear_btn.setEnabled(has_results)
    
    def start_crawling(self):
        """Start crawling"""
        urls_text = self.url_input.toPlainText().strip()
        if not urls_text:
            QMessageBox.warning(self, "Input Error", "Please enter URLs to crawl")
            return
        
        # Parse URLs
        urls = []
        for line in urls_text.split('\n'):
            line = line.strip()
            if line:
                if ',' in line:
                    urls.extend([url.strip() for url in line.split(',') if url.strip()])
                else:
                    urls.append(line)
        
        if not urls:
            QMessageBox.warning(self, "Input Error", "No valid URLs found")
            return
        
        # Configuration parameters
        config = {
            'max_depth': 3,
            'timeout': 30,
            'magic': True,
            'screenshot': False,
            'cache_mode': 1,  # 1=memory cache
            'headful': False,  # Default headless mode
            'verbose': True,
            'exclude_urls': [],  # Default value: empty list
            'user_agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",  # Default value
            'proxy': None,  # Default value: no proxy
            'concurrency': 3,  # Default value: concurrency is 3
            'css_selectors': None,  # Default value: no CSS selectors
            'extract_markdown': True,
            'extract_html': True,
            'generate_metadata': True
        }
        
        # Update UI status
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Crawling...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        # Clear previous results
        self.results = []
        self.update_results_table()
        
        # Start asynchronous crawling
        self.status_label.setText(f"Starting to crawl {len(urls)} URLs...")
        
        # Use QTimer to run asynchronous task
        QTimer.singleShot(100, lambda: self.run_async_task(urls, config))

    def start_generate_pipeline(self):
        """One-click training data generation: crawl or process PDF first then generate"""
        # Check if user has entered model and api_key
        provider = self.model_input.text().strip()
        api_key = self.api_key_input.text().strip()
        
        if not provider:
            QMessageBox.warning(self, "Input Error", "Please enter model name")
            return
            
        if not api_key:
            QMessageBox.warning(self, "Input Error", "Please enter API key")
            return
        
        # Check which tab is active
        current_tab = self.source_tabs.currentIndex()
        
        if current_tab == 0:  # Web crawling tab
            self.auto_generate_after_crawl = True
            self.start_crawling()
        elif current_tab == 1:  # PDF processing tab
            if not self.pdf_files:
                QMessageBox.warning(self, "No PDF Files", "Please select PDF files to process")
                return
            
            self.auto_generate_after_crawl = True
            self.start_pdf_processing()
    
    def run_async_task(self, urls, config):
        """Run asynchronous crawling task"""
        # Asyncio/threading has been moved to the top of the file
        
        async def run_crawl():
            await self.worker.run_crawl(urls, config)
        
        def run_in_thread():
            # Create new event loop in new thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(run_crawl())
            finally:
                loop.close()
        
        # Run event loop in new thread
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
    
    def stop_crawling(self):
        """Stop crawling"""
        self.worker.stop()
        self.status_label.setText("Stopping crawl...")
        self.stop_btn.setEnabled(False)
    
    def update_progress(self, message, progress):
        """Update progress"""
        self.status_label.setText(message)
        self.progress_bar.setValue(progress)
    
    def on_crawl_finished(self, result):
        """Crawling completion callback"""
        crawl_results = result['results']
        self.results.extend(crawl_results)  # Add crawl results to existing results
        self.status_label.setText(f"Crawling completed! Success: {sum(1 for r in crawl_results if r.get('success'))}/{len(crawl_results)}")
        self.progress_bar.setValue(100)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # Update results table
        self.update_results_table()
        
        # Only enable clear button (export training data enabled after generation)
        self.clear_btn.setEnabled(True)
        
        # Update LLM button status
        self.update_llm_buttons()

        # If one-click process, automatically generate training data
        if getattr(self, 'auto_generate_after_crawl', False):
            self.auto_generate_after_crawl = False
            # Clear selection to ensure all URLs are processed
            self.results_table.clearSelection()
            self.generate_training_data()
    
    def update_results_table(self):
        """Update results table"""
        try:
            # Temporarily disconnect selection signal to avoid triggering events during update
            self.results_table.itemSelectionChanged.disconnect(self.on_result_selected)
            signal_disconnected = True
        except:
            signal_disconnected = False
        
        try:
            self.results_table.setRowCount(len(self.results))
            
            for row, result in enumerate(self.results):
                # Status
                status_item = QTableWidgetItem("✅" if result.get('success') else "❌")
                status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if result.get('success'):
                    status_item.setForeground(QColor("#4caf50"))
                else:
                    status_item.setForeground(QColor("#f44336"))
                self.results_table.setItem(row, 0, status_item)
                
                # URL
                url_item = QTableWidgetItem(result.get('url', ''))
                self.results_table.setItem(row, 1, url_item)
                
                # Title
                title = result.get('title', 'N/A')
                if not result.get('success'):
                    title = result.get('error', 'Crawling failed')
                title_item = QTableWidgetItem(title)
                self.results_table.setItem(row, 2, title_item)
                
                # Time
                timestamp = result.get('timestamp', '')
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        time_str = dt.strftime("%H:%M:%S")
                    except:
                        time_str = timestamp.split('T')[1][:8] if 'T' in timestamp else timestamp
                else:
                    time_str = "N/A"
                time_item = QTableWidgetItem(time_str)
                self.results_table.setItem(row, 3, time_item)
        finally:
            # Reconnect signal
            if signal_disconnected:
                try:
                    self.results_table.itemSelectionChanged.connect(self.on_result_selected)
                except:
                    pass
    
    def on_result_selected(self):
        """Result selection changed"""
        try:
            selected_items = self.results_table.selectedItems()
            if not selected_items:
                return
            
            row = selected_items[0].row()
            if row < 0 or row >= len(self.results):
                return
            
            self.current_result = self.results[row]
            self.display_result_details()
        except Exception as e:
            print(f"Error selecting result: {e}")
            traceback.print_exc()
    
    def display_result_details(self):
        """Display result details"""
        try:
            if not self.current_result:
                return
            # Preview area removed, no longer displaying Markdown/HTML/metadata
            return
        except Exception as e:
            print(f"Error displaying result details: {e}")
            traceback.print_exc()
    
    def update_llm_buttons(self):
        """Update LLM related button status"""
        has_valid_results = any(r.get('success') and r.get('markdown') for r in self.results)
        if hasattr(self, 'convert_btn'):
            self.convert_btn.setEnabled(True)
    
    def generate_training_data(self):
        """Generate training data"""
        try:
            if not self.results:
                QMessageBox.warning(self, "Error", "No crawling results available for generating training data")
                return
            
            # Get selected results
            selected_indices = []
            selected_items = self.results_table.selectedItems()
            if selected_items:
                # Get selected rows
                selected_rows = set()
                for item in selected_items:
                    selected_rows.add(item.row())
                selected_indices = list(selected_rows)
                
                # If no rows selected, use all successful results
                if not selected_rows:
                    selected_indices = [i for i, r in enumerate(self.results) if r.get('success') and r.get('markdown')]
            else:
                # No specific rows selected, use all successful results
                selected_indices = [i for i, r in enumerate(self.results) if r.get('success') and r.get('markdown')]
            
            if not selected_indices:
                QMessageBox.warning(self, "No Valid Data", "No valid crawling results to process")
                return
            
            # Check if there are multiple URLs (more than one)
            if len(selected_indices) > 1:
                # Process each URL separately
                self.process_each_url_separately(selected_indices)
            else:
                # Process as before for single URL
                self.process_single_url(selected_indices)
            
        except Exception as e:
            error_msg = f"Unexpected error generating training data: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            QMessageBox.critical(self, "Error", error_msg)
            self.export_llm_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.status_label.setText("Processing failed")
    
    def process_each_url_separately(self, selected_indices):
        """Process each URL separately with individual LLM calls"""
        try:
            # Configure LLM (default + environment variables)
            provider, api_key, base_url, system_prompt, prompt_template, extra_params = self.get_default_llm_config()
            if not api_key:
                QMessageBox.warning(self, "API Key Missing", "No API key detected in environment variables, please set DASHSCOPE_API_KEY")
                return
            
            # Update UI status
            self.export_llm_btn.setEnabled(False)
            self.status_label.setText(f"Generating training data for {len(selected_indices)} URLs separately...")
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            
            # Store the indices for later use in callbacks
            self.selected_indices = selected_indices
            self.all_training_data = []
            self.current_url_index = 0
            
            # Start processing the first URL
            self.process_next_url()
            
        except Exception as e:
            error_msg = f"Error in process_each_url_separately: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            QMessageBox.critical(self, "Error", error_msg)
            self.export_llm_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.status_label.setText("Processing failed")
    
    def process_next_url(self):
        """Process the next URL in the list"""
        try:
            if self.current_url_index >= len(self.selected_indices):
                # All URLs have been processed
                self.on_all_urls_processed()
                return
            
            # Get the current URL to process
            idx = self.selected_indices[self.current_url_index]
            result = self.results[idx]
            
            if not result.get('success') or not result.get('markdown'):
                # Skip invalid results and move to the next
                self.current_url_index += 1
                self.process_next_url()
                return
            
            # Update progress
            progress = int((self.current_url_index / len(self.selected_indices)) * 80)  # Use 80% for processing
            self.progress_bar.setValue(progress)
            self.status_label.setText(f"Processing URL {self.current_url_index + 1}/{len(self.selected_indices)}: {result.get('title', 'Untitled')}")
            
            # Configure LLM
            provider, api_key, base_url, system_prompt, prompt_template, extra_params = self.get_default_llm_config()
            
            # Prepare content for the current URL
            content_item = {
                'url': result.get('url', ''),
                'title': result.get('title', ''),
                'content': result.get('markdown', '')
            }
            
            # Disconnect previous worker signals if they exist
            if hasattr(self, 'llm_worker'):
                try:
                    self.llm_worker.finished.disconnect()
                    self.llm_worker.error.disconnect()
                except:
                    pass
            
            # Create a new LLM worker for this URL
            self.llm_worker = LLMWorker()
            self.llm_worker.finished.connect(self.on_single_url_processed)
            self.llm_worker.error.connect(self.on_single_url_error)
            
            # Start processing this URL
            self.llm_worker.process_contents(
                contents=[content_item],  # Only one content item per call
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                system_prompt=system_prompt,
                prompt_template=prompt_template,
                extra_params=extra_params
            )
            
        except Exception as e:
            error_msg = f"Error in process_next_url: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            QMessageBox.critical(self, "Error", error_msg)
            self.export_llm_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.status_label.setText("Processing failed")
    
    def process_single_url(self, selected_indices):
        """Process a single URL as before"""
        try:
            # Prepare content to process
            contents_to_process = []
            for idx in selected_indices:
                result = self.results[idx]
                if result.get('success') and result.get('markdown'):
                    contents_to_process.append({
                        'url': result.get('url', ''),
                        'title': result.get('title', ''),
                        'content': result.get('markdown', '')  # Remove length limit, let large model process complete content
                    })
            
            if not contents_to_process:
                QMessageBox.warning(self, "No Valid Content", "No usable content in selected results")
                return
            
            # Configure LLM (default + environment variables)
            provider, api_key, base_url, system_prompt, prompt_template, extra_params = self.get_default_llm_config()
            if not api_key:
                QMessageBox.warning(self, "API Key Missing", "No API key detected in environment variables, please set DASHSCOPE_API_KEY")
                return
            
            # Update UI status
            self.export_llm_btn.setEnabled(False)
            self.status_label.setText("Generating training data using LLM...")
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            
            # Start processing
            try:
                # Connect the callback for single URL processing
                self.llm_worker.finished.connect(self.on_training_data_generated)
                self.llm_worker.process_contents(
                    contents=contents_to_process,
                    provider=provider,
                    api_key=api_key,
                    base_url=base_url,
                    system_prompt=system_prompt,
                    prompt_template=prompt_template,
                    extra_params=extra_params
                )
            except Exception as e:
                error_msg = f"Error starting processing: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                QMessageBox.critical(self, "Error", error_msg)
                self.export_llm_btn.setEnabled(True)
                self.progress_bar.setVisible(False)
                self.status_label.setText("Processing startup failed")
        except Exception as e:
            error_msg = f"Error in process_single_url: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            QMessageBox.critical(self, "Error", error_msg)
            self.export_llm_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.status_label.setText("Processing failed")
    
    def on_single_url_processed(self, result):
        """Callback for when a single URL processing is completed"""
        try:
            # Add the result to our collection
            url_data = result.get('data', [])
            self.all_training_data.extend(url_data)
            
            # Disconnect the signals from the current worker
            try:
                self.llm_worker.finished.disconnect()
                self.llm_worker.error.disconnect()
            except:
                pass
            
            # Move to the next URL
            self.current_url_index += 1
            
            # Process the next URL
            self.process_next_url()
            
        except Exception as e:
            error_msg = f"Error processing single URL result: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            QMessageBox.critical(self, "Error", error_msg)
            self.export_llm_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.status_label.setText("Processing failed")
    
    def on_all_urls_processed(self):
        """Callback for when all URLs have been processed"""
        try:
            # Update progress to 100%
            self.progress_bar.setValue(100)
            self.status_label.setText(f"Successfully generated training data from {len(self.selected_indices)} URLs")
            
            # Set the training data
            self.training_data = self.all_training_data
            
            # Update UI
            self.export_llm_btn.setEnabled(len(self.training_data) > 0)
            self.progress_bar.setVisible(False)
            
            # Preview data
            preview_data = self.training_data
            preview_text = json.dumps(preview_data, indent=2, ensure_ascii=False)
            self.training_data_preview.setPlainText(preview_text)
            
            # Show completion message
            QMessageBox.information(self, "Processing Complete", 
                                  f"Successfully generated {len(self.training_data)} training data entries from {len(self.selected_indices)} URLs")
            
        except Exception as e:
            error_msg = f"Error in on_all_urls_processed: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            QMessageBox.critical(self, "Error", error_msg)
            self.export_llm_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.status_label.setText("Processing failed")
    
    def on_single_url_error(self, error_msg):
        """Callback for when a single URL processing fails"""
        print(f"Error processing URL: {error_msg}")
        
        # Disconnect the signals from the current worker
        try:
            self.llm_worker.finished.disconnect()
            self.llm_worker.error.disconnect()
        except:
            pass
        
        # Move to the next URL even if the current one failed
        self.current_url_index += 1
        self.process_next_url()

    def get_default_llm_config(self):
        """Get default LLM configuration"""
        # Get model and api_key from user input
        provider = self.model_input.text().strip()
        api_key = self.api_key_input.text().strip()
        
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        system_prompt = "You are a data conversion assistant specializing in converting webpage content into high-quality training data."
        prompt_template = self.prompt_template
        extra_params = ""
        return provider, api_key, base_url, system_prompt, prompt_template, extra_params
    
    def on_training_data_generated(self, result):
        """Training data generation completed"""
        self.training_data = result.get('data', [])
        self.status_label.setText(f"Successfully generated {len(self.training_data)} training data entries")
        self.progress_bar.setValue(100)
        self.export_llm_btn.setEnabled(len(self.training_data) > 0)
        self.progress_bar.setVisible(False)
        
        # Preview data
        preview_data = self.training_data
        preview_text = json.dumps(preview_data, indent=2, ensure_ascii=False)
        self.training_data_preview.setPlainText(preview_text)
        
        # Disconnect the signal to avoid multiple connections
        try:
            self.llm_worker.finished.disconnect()
        except:
            pass
    
    def export_training_data(self):
        """Export training data"""
        if not self.training_data:
            QMessageBox.warning(self, "Export Error", "No training data available for export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Training Data",
            f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json);;CSV Files (*.csv)"
        )
        
        if not file_path:
            return
        
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.training_data, f, indent=2, ensure_ascii=False)
            else:
                # CSV export
                # Pandas has been moved to the top of the file
                df = pd.json_normalize(self.training_data)
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            QMessageBox.information(self, "Export Successful", f"Training data successfully exported to:\n{file_path}")
            self.status_label.setText(f"Exported {len(self.training_data)} training data entries")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error during export:\n{str(e)}")
    
    def export_results(self):
        """Export results"""
        if not self.results:
            QMessageBox.warning(self, "Export Error", "No results available for export")
            return
        
        file_path, file_type = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            f"crawl_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "CSV Files (*.csv);;JSON Files (*.json);;Excel Files (*.xlsx)"
        )
        
        if not file_path:
            return
        
        try:
            if file_type == "CSV Files (*.csv)":
                self.export_to_csv(file_path)
            elif file_type == "JSON Files (*.json)":
                self.export_to_json(file_path)
            elif file_type == "Excel Files (*.xlsx)":
                self.export_to_excel(file_path)
            
            QMessageBox.information(self, "Export Successful", f"Results successfully exported to:\n{file_path}")
            self.status_label.setText(f"Exported {len(self.results)} results")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error during export:\n{str(e)}")
    
    def export_to_csv(self, file_path):
        """Export to CSV"""
        # CSV has been moved to the top of the file
        
        with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['URL', 'Title', 'Status', 'Time', 'Markdown Content'])
            
            for result in self.results:
                writer.writerow([
                    result.get('url', ''),
                    result.get('title', ''),
                    'Success' if result.get('success') else 'Failed',
                    result.get('timestamp', ''),
                    (result.get('markdown', '')[:500] + '...') if result.get('markdown') else ''  # Truncate long content
                ])
    
    def export_to_json(self, file_path):
        """Export to JSON"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
    
    def export_to_excel(self, file_path):
        """Export to Excel"""
        try:
            # Convert to DataFrame
            df_data = []
            for result in self.results:
                df_data.append({
                    'URL': result.get('url', ''),
                    'Title': result.get('title', ''),
                    'Status': 'Success' if result.get('success') else 'Failed',
                    'Time': result.get('timestamp', ''),
                    'Markdown Preview': (result.get('markdown', '')[:200] + '...') if result.get('markdown') else ''
                })
            
            df = pd.DataFrame(df_data)
            df.to_excel(file_path, index=False)
            
        except ImportError:
            raise Exception("Missing pandas library. Please install: pip install pandas openpyxl")
    
    def clear_results(self):
        """Clear results"""
        reply = QMessageBox.question(self, "Confirm Clear",
                                    "Are you sure you want to clear all results?",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            # Clear URL input box
            self.url_input.clear()
            
            # Clear PDF files list
            self.pdf_files = []
            self.update_pdf_list_display()
            
            # Clear crawling results
            self.results = []
            self.current_result = None
            self.update_results_table()
            
            # Clear training data
            self.training_data = []
            self.training_data_preview.clear()
            
            # Update button status
            self.clear_btn.setEnabled(False)
            self.export_llm_btn.setEnabled(False)
            self.clear_pdf_results_btn.setEnabled(False)
            self.status_label.setText("Results cleared")
            self.update_llm_buttons()
    
    def show_error(self, message):
        """Show error"""
        QMessageBox.critical(self, "Error", message)
        self.status_label.setText(f"Error: {message}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
    
    def closeEvent(self, event):
        """Close event"""
        # Stop worker threads
        self.worker.stop()
        self.llm_worker.stop()
        self.pdf_worker.stop()
        
        # Wait for threads to finish
        self.thread.quit()
        self.llm_thread.quit()
        self.pdf_thread.quit()
        self.thread.wait()
        self.llm_thread.wait()
        self.pdf_thread.wait()
        
        event.accept()

def main():
    """Main function"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    app.setApplicationName("CapraData")
    app.setApplicationDisplayName("CapraData")
    app.setWindowIcon(QIcon(resource_path("sign.png")))
    
    # Create palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(245, 245, 245))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(44, 62, 80))
    palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Text, QColor(44, 62, 80))
    palette.setColor(QPalette.ColorRole.Button, QColor(74, 111, 165))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(33, 150, 243))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    # Create and show main window
    window = ModernCrawlerUI()
    window.setWindowIcon(QIcon(resource_path("sign.png")))
    window.showMaximized()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()