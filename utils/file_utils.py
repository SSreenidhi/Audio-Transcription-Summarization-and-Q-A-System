import os

def read_text_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def write_text_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def file_exists(filepath):
    return os.path.exists(filepath)