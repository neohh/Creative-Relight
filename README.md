# Creative Relight

Windows-приложение для ИИ-обработки изображений: генерирует карты **Albedo, Shading, Specular, Depth, Normal** из одного входного изображения (PyQt6 GUI, PyTorch CUDA).

AI-powered relighting tool: generates **Albedo / Shading / Specular / Depth / Normal** maps from a single input image (PyQt6 GUI, PyTorch CUDA).

## Релизы / Releases

| Версия | Что нового |
|--------|-----------|
| **v1.0.1** | Оригинальная версия (CUDA 12.8 build) / Original build |
| **v2.0.0** | **Поддержка EXR/HDR входа** — прозрачные `.exr` спрайты: линейные float данные тонмапятся в sRGB, альфа-канал сохраняется как `alpha/alpha_XXXXXX.png`, прозрачные области заполняются соседними цветами для моделей / **EXR/HDR input support** — transparent `.exr` sprites: linear float data is tonemapped to sRGB, the alpha channel is exported as an `alpha` mask, and fully transparent regions are inpainted from surrounding colors before inference |

## Форматы входа / Supported input

- PNG, JPEG, TIFF, BMP, GIF, WEBP
- **v2.0.0+:** EXR (OpenEXR, float16/float32, RGBA), HDR (Radiance), PIC

## Установка / Install

1. Скачайте все части архива релиза и соберите: `cat CreativeRelight-*.tar.part-* > CreativeRelight.tar && tar -xf CreativeRelight.tar`
2. Запустите `Creative Relight.exe`
3. Модели скачаются автоматически при первом запуске (HuggingFace)

## Сборка из исходников / Building from source

Исходники приложения лежат в `src/` и `ui/`. Приложение поставляется как PyInstaller onedir-сборка; exe содержит замороженный PYZ-архив с модуями приложения. Инструменты для патча PYZ лежат в `tools/` (см. `tools/patch_exe.py`).

App sources are in `src/` and `ui/`. The app ships as a PyInstaller onedir build; the exe embeds a frozen PYZ archive with the app modules. PYZ surgery tools are in `tools/`.

## Структура / Layout

```
Creative Relight.exe      — основной exe (PyInstaller, Python 3.13)
_internal/                — runtime (torch CUDA, PyQt6, модели) — в архивах релиза
src/                      — исходники процессоров и утилит
ui/                       — исходники интерфейса (PyQt6)
config.yaml               — конфигурация загрузки моделей
```
