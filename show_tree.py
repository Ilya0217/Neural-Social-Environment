#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Генерация дерева проекта с исключениями."""

import sys
from pathlib import Path

# Устанавливаем UTF-8 для вывода
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def tree(path, prefix='', exclude_dirs=None, exclude_files=None, exclude_ext=None, except_files=None):
    """Рекурсивно выводит дерево директорий."""
    if exclude_dirs is None:
        exclude_dirs = {'__pycache__', 'uml_diagrams', '.git', '.venv', 'venv'}
    if exclude_files is None:
        exclude_files = {'generate_uml.py', 'generate_uml_simple.py', 'show_tree.py', 'project_tree.txt'}
    if exclude_ext is None:
        exclude_ext = {'.md'}
    if except_files is None:
        except_files = {'README.md'}
    
    path = Path(path)
    
    # Пропускаем исключённые директории
    if path.name in exclude_dirs:
        return
    
    # Получаем элементы, исключая ненужные
    items = []
    for p in sorted(path.iterdir()):
        if p.name in exclude_dirs:
            continue
        if p.is_file():
            # Исключаем файлы по имени
            if p.name in exclude_files:
                continue
            # Исключаем файлы по расширению, кроме исключений
            if p.suffix in exclude_ext and p.name not in except_files:
                continue
        items.append(p)
    
    # Разделяем на директории и файлы (директории сначала)
    dirs = sorted([p for p in items if p.is_dir()], key=lambda x: x.name.lower())
    files = sorted([p for p in items if p.is_file()], key=lambda x: x.name.lower())
    all_items = dirs + files
    
    # Выводим элементы
    for i, item in enumerate(all_items):
        is_last = i == len(all_items) - 1
        current_prefix = '└── ' if is_last else '├── '
        print(prefix + current_prefix + item.name)
        
        if item.is_dir():
            next_prefix = prefix + ('    ' if is_last else '│   ')
            tree(item, next_prefix, exclude_dirs, exclude_files, exclude_ext, except_files)

if __name__ == "__main__":
    print("agent_dialogue_sim/")
    tree('.')

