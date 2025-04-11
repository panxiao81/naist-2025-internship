# from data_cleaning import lists, convert_md
# from datasets import Dataset

# input_folder = 'docs'     # Markdown 文件所在目录
# output_folder = 'output_json_files'  # 清洗后的文本输出目录

# convert_md.process_markdown_folder(input_folder, output_folder)

# result: list = lists.read_and_flatten_json_arrays("output_json_files")

# doc_dataset = Dataset.from_list(result)

# doc_dataset.to_parquet("document-dataset.parquet")

# Thanks to ChatGPT 4-o, It generates fully functional code.

import os
import re
import json

def split_markdown_paragraphs(md_text):
    """
    分段 Markdown 文本，同时保留代码块整体，不做分段处理。
    """
    paragraphs = []
    lines = md_text.splitlines()
    in_code_block = False
    buffer = []

    for line in lines:
        if line.strip().startswith("```"):
            if in_code_block:
                buffer.append(line)
                paragraphs.append('\n'.join(buffer).strip())
                buffer = []
                in_code_block = False
            else:
                if buffer:
                    paragraphs.append('\n'.join(buffer).strip())
                    buffer = []
                in_code_block = True
                buffer.append(line)
        elif in_code_block:
            buffer.append(line)
        elif line.strip() == '':
            if buffer:
                paragraphs.append('\n'.join(buffer).strip())
                buffer = []
        else:
            buffer.append(line)

    if buffer:
        paragraphs.append('\n'.join(buffer).strip())

    return paragraphs

def clean_mdx_jsx(content):
    """
    移除 .mdx 文件中的 JSX 元素，仅保留纯 Markdown。
    这使用一个简单的正则移除形如 <...>...</...> 和自闭合 <.../> 标签。
    """
    # 移除多行标签如 <Component>...</Component>
    content = re.sub(r'<[^>]+>.*?</[^>]+>', '', content, flags=re.DOTALL)
    # 移除单行自闭合标签 <Component />
    content = re.sub(r'<[^>/]+/?>', '', content)
    return content

def process_markdown_file(input_path, output_path):
    """
    处理单个 Markdown 或 MDX 文件，保存输出为 JSONL 格式。
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if input_path.endswith('.mdx'):
        content = clean_mdx_jsx(content)

    paragraphs = split_markdown_paragraphs(content)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for para in paragraphs:
            if para.strip():
                json.dump({"text": para.strip()}, out_f, ensure_ascii=False)
                out_f.write('\n')

def process_directory(input_dir, output_dir):
    """
    递归处理目录中的所有 .md 和 .mdx 文件，并输出到指定目录。
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.md') or file.endswith('.mdx'):
                rel_dir = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, rel_dir)
                output_filename = file.rsplit('.', 1)[0] + '.jsonl'
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_subdir, output_filename)
                process_markdown_file(input_path, output_path)
                print(f"Processed: {input_path} → {output_path}")

input_dir = "docs"
output_dir = "output-jsonl-files"
process_directory(input_dir, output_dir)

from datasets import Dataset

def collect_jsonl_texts(directory):
    all_texts = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if "text" in data and isinstance(data["text"], str):
                                all_texts.append({"text": data["text"]})
                        except json.JSONDecodeError:
                            print(f"警告: 无法解析 {file_path} 中的一行内容: {line}")
    return all_texts

def create_and_save_dataset(data_list, output_path):
    dataset = Dataset.from_list(data_list)
    dataset.save_to_disk(output_path)
    print(f"✅ 数据集已保存至: {output_path}（共 {len(dataset)} 条记录）")

# 示例路径（你可以改为自己的路径）
input_directory = "output-jsonl-files"     # 输入目录
output_dataset_path = "md-version-dataset"  # 输出目录（非文件！）

# 主流程
texts = collect_jsonl_texts(input_directory)
create_and_save_dataset(texts, output_dataset_path)
