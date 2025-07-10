# HTML→Markdown Converter (Google Engineer Optimized Version)

A professional, highly configurable HTML to Markdown converter with a GUI, offering:
- Multiple conversion engines (Readability, Crawl4AI, Jina, html-to-markdown)
- Advanced Markdown cleaning options (remove short lines, contract empty lines, de-duplicate headlines, etc.)
- Quality analysis: fidelity, ad retention, word count, and more
- User-friendly Tkinter GUI with real-time progress and detailed result reports
- Settings panel for Markdown cleaning with live preview
- Supports direct URL, local file, or HTML input

## Features

- **Conversion Engines:** Choose between Readability, Crawl4AI, Jina (for URLs), or html-to-markdown.
- **Markdown Cleaning:** Fine-tune how Markdown output is cleaned and compressed.
- **Quality Metrics:** Analyze conversion quality with fidelity and ad retention scores.
- **GUI:** Intuitive Tkinter interface, including result previews and batch task queue.
- **Save Results:** Export cleaned markdown easily.

## Installation

1. Clone this repository.
    ```bash
    git clone https://github.com/021gink/html2md.git
    ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   python main.py
   ```



## Usage

- **Input:** Paste a URL, HTML text, or select a local HTML file.
- **Select Engine:** Use the combobox to pick the converter engine.
- **Settings:** Click “清洗设置” (Clean Settings) to adjust cleaning options and preview results.
- **Convert:** Click “开始转换” (Start Conversion).
- **Results:** Switch between cleaned Markdown, original Markdown, and cleaned HTML. Analyze quality metrics in the “分析报告” (Analysis Report) tab.
- **Export:** Save the cleaned Markdown using “保存 Markdown...” (Save Markdown).

## Dependencies

- `beautifulsoup4`
- `crawl4ai`
- `readabilipy`
- `lxml`
- `httpx`
- `requests`
- `html-to-markdown`
- `markdowncleaner`

## Notes

- The Jina engine only supports URL input.
- Configuration is persisted in `converter_config.json`.
- The application is cross-platform (Windows, macOS, Linux).


## Acknowledgements

- [readabilipy](https://github.com/alan-turing-institute/readabilipy)
- [crawl4ai](https://github.com/datawhalechina/crawl4ai)
- [markdowncleaner](https://github.com/0x7d8/markdowncleaner)
- [html-to-markdown](https://github.com/Alir3z4/html2text)