# Face_detection
# Yüz Bulanıklaştırma Uygulaması

Bu proje Python, OpenCV ve MediaPipe kullanarak resim, video veya canlı webcam görüntüsünde yüzleri tespit eder ve tespit edilen yüzleri bulanıklaştırır.

## Gereksinimler

- Python 3.x
- OpenCV
- MediaPipe

## Kurulum

Gerekli kütüphaneleri aşağıdaki komutla yükleyebilirsiniz:


## Kullanım

Programı komut satırından şu şekilde çalıştırabilirsiniz:


- `--mode` parametresi:
  - `image` — Resim üzerinde çalışır (filePath gerekli)
  - `video` — Video üzerinde çalışır (filePath gerekli)
  - `webcam` — Canlı webcam görüntüsünü kullanır (filePath gerekmez)

- `--filePath` parametresi, işlenecek resim veya video dosyasının yoludur (sadece `image` ve `video` modlarında gereklidir).

## Çıktılar

- İşlenmiş dosyalar `output` klasörüne kaydedilir.
- Webcam modunda sonuç gerçek zamanlı olarak ekranda gösterilir.

## Yazar

Badalov Abulfat

---

Teşekkürler!
