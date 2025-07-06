# whisper_transcribe.py — Script for batch audio/video transcription using faster-whisper
# Licensed under MIT License
# Copyright (c) 2025 Dmitrii Shamaraev  -> https://t.me/turmerig -> Telegram @turmerig

"""
✨ Скрипт автоматической транскрибации аудио/видео файлов с помощью faster-whisper

📌 Описание:
Этот скрипт предназначен для массовой расшифровки аудио и видео файлов.
Поддерживаются форматы: .mp4, .avi, .mov, .mkv, .wav, .mp3, .ogg, .flac

🔹 Функционал:
- Обработка всех файлов из указанной папки
- Автоматическое определение модели Whisper на основе доступной RAM
- Расшифровка на русском языке с поддержкой пользовательских слов (hotwords)
- Копирование исходных файлов в выходную папку
- Подсчёт времени обработки каждого файла и общего времени работы
- Расчёт скорости расшифровки: во сколько раз быстрее, чем реальное время
- Генерация информационного файла _transcription.info с метаданными
- Всплывающее уведомление о завершении (Windows)

📦 Требуемые библиотеки:
pip install faster-whisper moviepy winotify psutil

📂 Структура выходной папки:
- *.txt — текстовые файлы с транскрипцией
- *.mp4 / *.wav и др. — копии исходных медиа-файлов
- _transcription.info — подробный отчёт по всем файлам и параметрам

⚙️ Параметры запуска:
- python whisper_transcribe.py                  → обрабатывает все файлы из папки `_input`
- python whisper_transcribe.py my_media         → обрабатывает файлы из папки `my_media`
- python whisper_transcribe.py my_media --model small → использует модель `small`

🕒 Пример вывода времени:
⏱ Время обработки: 245.67 сек (04:05)
⚡ Скорость расшифровки: 6.5x → файл длиной 60 секунд обработан за ~9 секунд

🔔 Уведомления:
На Windows показывает всплывающие уведомления о завершении задачи (если установлен winotify)

📊 Отчёт:
Сохраняет информацию в файл `_transcription.info`:
- Версия скрипта
- Информация о каждом файле (путь, длительность, размер, время обработки, скорость)
- Системная информация (RAM, CUDA, версии библиотек)
- Общее время работы и средняя скорость расшифровки
"""
# TODO v24
SCRIPT_VERSION = 'v24'

import argparse
import datetime
import os
import psutil
import shutil
import time
from faster_whisper import WhisperModel
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip


def get_available_ram():
    """
    Возвращает доступную оперативную память в ГБ
    """
    return psutil.virtual_memory().available / (1024 ** 3)


def has_cuda():
    """
    Проверяет, доступна ли CUDA (требуется torch)
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def suggest_model(ram_gb, use_cuda):
    """
    Возвращает рекомендуемую модель в зависимости от ресурсов
    """
    if use_cuda:
        return "large"
    elif ram_gb >= 6:
        return "small"
    elif ram_gb >= 4:
        return "base"
    else:
        return "tiny"


def extract_audio_from_video(video_path, audio_path):
    """
    Извлекает аудиодорожку из видео и сохраняет в WAV
    """
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec='pcm_s16le')
    return audio_path


def is_supported_file(filename):
    """
    Проверяет, является ли файл поддерживаемым аудио/видео форматом
    """
    supported_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wav", ".mp3", ".ogg", ".flac"]
    return os.path.isfile(filename) and os.path.splitext(filename)[1].lower() in supported_extensions


def get_media_duration(file_path):
    """
    Возвращает длительность аудио/видео файла в формате MM:SS
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".mp4", ".avi", ".mov", ".mkv"):
            clip = VideoFileClip(file_path)
        elif ext in (".wav", ".mp3", ".ogg", ".flac"):
            clip = AudioFileClip(file_path)
        else:
            return "не определено"
        total_seconds = int(clip.duration)
        minutes, seconds = divmod(total_seconds, 60)
        return f"{minutes:02d}:{seconds:02d}"
    except Exception as e:
        print(f"❌ [WARNING] Не удалось получить длительность файла '{file_path}': {e}")
        return "ошибка"


def read_hotwords(file_path='hotwords.txt'):
    """
    Читает список hotwords из текстового файла
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            return ', '.join(lines) if lines else None
    except FileNotFoundError:
        print(f"[WARNING] Файл hotwords.txt не найден. Используются значения по умолчанию")
        return ''


def duration_to_seconds(duration_str):
    """
    Преобразует строку в формате 'MM:SS' в количество секунд
    """
    try:
        minutes, seconds = map(int, duration_str.split(":"))
        return minutes * 60 + seconds
    except Exception:
        return 0  # если не удалось определить длительность


def format_seconds(seconds):
    """
    Преобразует секунды в формат HH:MM:SS или MM:SS
    """
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


def transcribe_file(hotwords, media_path, output_folder, model_size="base"):
    """
    Расшифровывает указанный файл (аудио/видео) и возвращает транскрибацию + время выполнения
    media_path - путь к исходному файлу
    output_folder - папка, куда можно положить временный аудиофайл
    model_size - размер модели Whisper
    """
    start_time = time.time()  # Начало отсчёта
    # Определяем расширение файла
    ext = os.path.splitext(media_path)[1].lower()
    # Если это видео — извлекаем аудио
    if ext in (".mp4", ".avi", ".mov", ".mkv"):
        print("[INFO] Обнаружено видео. Извлечение аудио...")
        file_root = os.path.splitext(os.path.basename(media_path))[0]
        audio_path = os.path.join(output_folder, f"{file_root}_temp_audio.wav")
        audio_path = extract_audio_from_video(media_path, audio_path)
    # Если это аудио — работаем напрямую
    elif ext in (".wav", ".mp3", ".ogg", ".flac"):
        audio_path = media_path
    # Неподдерживаемый формат
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {media_path}")
    # Загрузка модели Whisper
    print(f"[INFO] Загрузка модели Whisper ({model_size})...")
    model = WhisperModel(model_size, device="cuda" if has_cuda() else "cpu")
    # Расшифровка
    print("[INFO] Начинаю расшифровку...")
    # Получаем сегменты и информацию о языке и др.
    # segments, info = model.transcribe(audio_path, beam_size=5)
    # segments, info = model.transcribe(audio_path, beam_size=5, language="ru")
    segments, info = model.transcribe(
        audio_path,
        language="ru",
        log_progress=True,
        hotwords=hotwords,
    )
    print(f"[INFO] Обнаруженный язык: {info.language}")
    # Собираем все сегменты в одну строку без лишних переносов
    transcription = " ".join(segment.text.strip() for segment in segments).strip()
    # TODO Удаляем временный аудиофайл, если он был создан
    # if "_temp_audio.wav" in audio_path and os.path.exists(audio_path):
    #     os.remove(audio_path)
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    return transcription, elapsed_time  # возвращаем транскрибацию и время


def get_system_info():
    """
    Возвращает системную информацию
    """
    import sys
    import platform
    return {
        "os": platform.platform(),
        "python_version": sys.version.split("|")[0].strip(),
        "cuda_available": has_cuda(),
        "ram_available_gb": round(get_available_ram(), 2),
    }


def get_library_versions():
    """
    Возвращает версии используемых библиотек
    """
    from importlib.metadata import version
    libraries = {
        "faster-whisper": None,
        "whisper": None,
        "torch": None,
        "moviepy": None,
        "winotify": None,
        "argparse": None,
        "psutil": None,
    }
    for lib in libraries:
        try:
            libraries[lib] = version(lib)
        except ImportError:
            libraries[lib] = "not installed"
        except Exception:
            libraries[lib] = "unknown"
    return libraries


def write_info_file(hotwords, output_folder, processed_info, failed_info, model_size, total_time_sec):
    """
    Записывает информацию обо всех обработанных и необработанных файлах
    """
    info_path = os.path.join(output_folder, "_transcription.info")
    system_info = get_system_info()
    library_versions = get_library_versions()
    # Параметры транскрибации
    transcribe_params = {
        "engine": "faster-whisper",
        "engine_version": library_versions.get("faster-whisper", "unknown"),
        "model_size": model_size,
        "language": "ru",
        "log_progress": True,
        "hotwords": hotwords,
    }
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("=== 🧾 Версия скрипта ==========\n")
        f.write(f"{SCRIPT_VERSION}\n\n")
        # Информация об успешных файлах
        f.write("=== 📁 Успешно обработанные файлы ==========\n")
        for info in processed_info:
            f.write(f"📌 Файл: {info['name']}\n")
            f.write(f"Путь: {info['path']}\n")
            f.write(f"Размер файла: {info['size_mb']} MB\n")
            f.write(f"Длительность: {info['duration']}\n")
            f.write(f"Время обработки: {info['elapsed_time_sec']:.2f} сек ({info['elapsed_time_str']})\n")
            f.write(f"Скорость расшифровки: {info['speed_x']:.2f}x\n")
            f.write("-" * 50 + "\n")
        # Информация об ошибках
        if failed_info:
            f.write("=== ❌ Файлы с ошибками ==========\n")
            for info in failed_info:
                f.write(f"📌 Файл: {info['name']}\n")
                f.write(f"Путь: {info['path']}\n")
                f.write(f"Ошибка: {info['reason']}\n")
                f.write("-" * 50 + "\n")
        # Параметры транскрибации
        f.write("\n\n=== ⚙️ Параметры транскрибации ==========\n")
        f.write(f"Библиотека: {transcribe_params.get('engine', '')} (v{transcribe_params.get('engine_version', '')})\n")
        f.write(f"Модель: {transcribe_params.get('model_size', '')}\n")
        for key, value in transcribe_params.items():
            if key not in ("model_size", "engine", "engine_version"):
                f.write(f"{key}: {value}\n")
        # Системная информация
        f.write("\n\n=== 💻 Системная информация ==========\n")
        for key, value in system_info.items():
            f.write(f"{key}: {value}\n")
        # Библиотеки и их версии
        f.write("\n\n=== 🧠 Библиотеки и их версии ==========\n")
        for lib, ver in library_versions.items():
            f.write(f"{lib}: {ver}\n")
        # Общее время работы
        f.write("\n\n=== 📊 Общая статистика ==========\n")
        if processed_info:
            total_media_duration = sum(f['media_duration_sec'] for f in processed_info)
            total_elapsed_time = sum(f['elapsed_time_sec'] for f in processed_info)
            average_speed = total_media_duration / total_elapsed_time if total_elapsed_time > 0 else 0
            f.write(f"Общая длительность медиа: {format_seconds(total_media_duration)}\n")
            f.write(f"Общее время обработки: {total_elapsed_time:.2f} сек ({format_seconds(total_elapsed_time)})\n")
            f.write(f"Средняя скорость расшифровки: {average_speed:.2f}x\n\n")
        else:
            f.write("Файлов успешно обработано: 0\n")
        f.write("=== ℹ️ Пояснение к скорости расшифровки ==========\n")
        f.write("Скорость расшифровки показывает, во сколько раз модель обработала файл быстрее реального времени.\n")
        f.write("Пример: 6.5x — файл длиной 60 секунд обработан за ~9 секунд.\n")
        f.write("Чем выше значение, тем эффективнее работает модель.\n")
        f.write(f"\nОбщее время работы скрипта: {total_time_sec:.2f} сек ({format_seconds(total_time_sec)})\n")
    print(f"\n[INFO] Информационный файл сохранён: {info_path}\n")


def show_notification(title, message='', duration=4):
    """
    Выводит всплывающее уведомление в Windows
    Если библиотека winotify не установлена — выводит в консоль
    """
    try:
        from winotify import Notification
        toast = Notification(
            app_id="Transcriber",
            title=title,
            msg=message,
            duration="short"
        )
        toast.show()
    except ImportError:
        print(f"[NOTIFY] {title}: {message}")


if __name__ == "__main__":
    start_total_time = time.time()
    print("\nПакетная расшифровка аудио/видео через Whisper")
    print(f"\n🧾 Версия скрипта {SCRIPT_VERSION}\n")
    parser = argparse.ArgumentParser(description="Пакетная расшифровка аудио/видео через Whisper")
    parser.add_argument("input_folder", type=str, nargs="?", default="_input",
                        help="Путь к папке с исходными аудио/видео файлами (по умолчанию: '_input')")
    parser.add_argument("--model", type=str, default=None,
                        choices=["tiny", "base", "small", "medium", "large",
                                 "tiny.en", "base.en", "small.en", "medium.en"],
                        help="Модель Whisper: tiny, base, small, medium, large (или .en для англоязычных). Если не указано — автоматически")
    args = parser.parse_args()
    input_folder = args.input_folder
    # Если папка не существует и это значение по умолчанию — создаём её
    if not os.path.exists(input_folder):
        print(f"[INFO] Папка '{input_folder}' не найдена. Создаём новую...")
        os.makedirs(input_folder)
    # Проверяем, что это действительно папка
    if not os.path.isdir(input_folder):
        raise NotADirectoryError(f"Указанный путь не является папкой: {input_folder}")
    # Получаем список всех подходящих файлов
    files_to_process = [
        os.path.join(input_folder, f) for f in os.listdir(input_folder)
        if is_supported_file(os.path.join(input_folder, f))
    ]
    if not files_to_process:
        print("❌ [WARNING] В указанной папке нет подходящих файлов для расшифровки")
        exit()
    print(f"[INFO] Найдено {len(files_to_process)} файлов для обработки")
    # Автоматический выбор модели на основе доступной памяти и CUDA
    ram_available = get_available_ram()
    cuda_available = has_cuda()
    chosen_model = args.model if args.model else suggest_model(ram_available, cuda_available)
    print(f"[INFO] Доступная RAM: {ram_available:.2f} ГБ")
    print(f"[INFO] Доступна CUDA: {'Да' if cuda_available else 'Нет'}")
    print(f"[INFO] Выбрана модель: {chosen_model}")
    # Создаём имя папки по текущей дате и времени + модель
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{current_time}_{chosen_model}"
    output_folder = os.path.join(os.getcwd(), folder_name)
    # Создаём папку, если её нет
    os.makedirs(output_folder, exist_ok=True)
    # Списки для хранения информации о файлах
    processed_files_info = []
    failed_info = []
    total_files = len(files_to_process)
    hotwords = read_hotwords()
    # Обрабатываем каждый файл по очереди
    for idx, media_file in enumerate(files_to_process, start=1):
        file_name = os.path.basename(media_file)
        file_root, _ = os.path.splitext(file_name)
        duration = get_media_duration(media_file)
        print(f"\n🔄 Обработка файла {idx} из {total_files}: {file_name} | Длительность: {duration}")
        # Копируем исходный файл в выходную папку
        try:
            shutil.copy(media_file, os.path.join(output_folder, file_name))
            print(f"📎 Файл скопирован в выходную папку")
        except Exception as e:
            print(f"[ERROR] Не удалось скопировать файл '{file_name}' в выходную папку: {str(e)}")
        if duration == "ошибка":
            print(f"❌ [ERROR] Не удалось определить длительность файла '{file_name}'. Возможно, файл повреждён")
            failed_info.append({
                'name': file_name,
                'path': media_file,
                'reason': 'Ошибка получения длительности'
            })
            continue  # Пропускаем такой файл
        try:
            transcription, elapsed_time = transcribe_file(hotwords, media_file, output_folder, model_size=chosen_model)
            # Формируем имя выходного файла
            output_file = os.path.join(output_folder, f"{file_root}.txt")
            # Сохраняем транскрибацию
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(transcription)
            # Добавляем информацию о времени в удобочитаемом формате
            elapsed_minutes = elapsed_time // 60
            elapsed_seconds = elapsed_time % 60
            print(f"✅ Транскрибация сохранена: {output_file}")
            print(f"⏱ Время обработки: {elapsed_time:.2f} сек ({format_seconds(elapsed_time)})")
            # Добавляем информацию о файле
            # Переводим длительность из MM:SS в секунды
            media_duration_sec = duration_to_seconds(duration)
            # Рассчитываем скорость расшифровки
            try:
                speed = media_duration_sec / elapsed_time
            except ZeroDivisionError:
                speed = 0
            print(f"⏱ Скорость расшифровки: {speed:.2f}x")
            processed_files_info.append({
                'name': file_name,
                'path': media_file,
                'duration': duration,
                'media_duration_sec': media_duration_sec,
                'size_mb': round(os.path.getsize(media_file) / (1024 ** 2), 2),
                'elapsed_time_sec': elapsed_time,
                'elapsed_time_str': format_seconds(elapsed_time),
                'speed_x': round(speed, 2)
            })
        except Exception as e:
            print(f"[ERROR] Не удалось обработать файл '{file_name}': {str(e)}")
            failed_info.append({
                'name': file_name,
                'path': media_file,
                'reason': str(e)
            })
    # Общее время работы
    end_total_time = time.time()
    total_elapsed_time_sec = end_total_time - start_total_time
    total_elapsed_time_str = format_seconds(total_elapsed_time_sec)
    print("\n✅🎉 Все файлы обработаны\n")
    print(f"⏱ Общее время работы скрипта: {total_elapsed_time_sec:.2f} сек ({total_elapsed_time_str})")
    if len(failed_info) > 0:
        error_failed_info = '\n❌'
    else:
        error_failed_info = ', '
    print(f"\n✅ 📊 Успешно: {len(processed_files_info)}{error_failed_info}Ошибок: {len(failed_info)}")
    if processed_files_info:
        total_media_duration = sum(f['media_duration_sec'] for f in processed_files_info)
        total_elapsed_time = sum(f['elapsed_time_sec'] for f in processed_files_info)
        average_speed = total_media_duration / total_elapsed_time if total_elapsed_time > 0 else 0
        print(f"📊 Общая длительность медиа: {format_seconds(total_media_duration)}")
        print(f"⏱ Общее время обработки: {total_elapsed_time:.2f} сек ({format_seconds(total_elapsed_time)})")
        print(f"⚡ Средняя скорость расшифровки: {average_speed:.2f}x")
        print("ℹ️ Пояснение:")
        print("   - 1.0 — обработка в реальном времени.")
        print("   - >1.0 — модель работает быстрее оригинала. Например:")
        print("     6.5x → файл длиной 60 секунд обработан за ~9 секунд.")
        print("   - <1.0 — обработка медленнее оригинала (чаще при высоком VAD/шумах).")
    if failed_info:
        print("\n❌ Файлы с ошибками:")
        for f in failed_info:
            print(f"- {f['name']} — {f['reason']}")
    # Сохраняем информационный файл
    write_info_file(hotwords, output_folder, processed_files_info, failed_info, chosen_model, total_elapsed_time_sec)
    # Показываем уведомление
    show_notification("\n✅ Готово!", f"Обработано {len(processed_files_info)} файлов. Результаты в папке:\n{output_folder}")
