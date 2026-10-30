"""Чтение исходных данных задания из JSON-файла."""

import argparse
import json
from pathlib import Path


def read_input(problem, argv=None):
    parser = argparse.ArgumentParser(description=f'Исходные данные задания {problem}')
    parser.add_argument(
        'input_file', nargs='?', type=Path,
        default=Path(__file__).resolve().parent / 'data' / f'problem{problem}.json',
        help='JSON-файл с данными (по умолчанию data/problemN.json рядом с программой)',
    )
    args = parser.parse_args(argv)
    try:
        with args.input_file.open(encoding='utf-8') as source:
            data = json.load(source)
    except (OSError, ValueError) as error:
        parser.error(f'Не удалось прочитать {args.input_file}: {error}')
    if not isinstance(data, dict):
        parser.error('В JSON должен быть объект с параметрами задания')
    required = {1: ('a', 'b'), 2: ('x', 'y'),
                3: ('a', 'b', 'epsilon', 'max_iterations'),
                4: ('a', 'epsilon', 'max_iterations'),
                5: ('a', 'epsilon', 'max_iterations')}[problem]
    missing = [name for name in required if name not in data]
    if missing:
        parser.error('В файле отсутствуют поля: ' + ', '.join(missing))
    return data
