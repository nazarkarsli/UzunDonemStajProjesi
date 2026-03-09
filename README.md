# UzunDonemStajProjesi
Video akışı üzerinden elde edilen görsel ve işitsel verileri işleyerek çok modlu (multimodal) duygu/stres analizi yapan ve şirket psikoloğu değerlendirme modülü içeren yapay zeka tabanlı uzun dönem (7+1) staj projesi.

# Çok Modlu Duygu ve Stres Analizi Sistemi

##  Proje Hakkında
Bu proje, Bilgisayar Mühendisliği 7+1 uzun dönem staj uygulaması kapsamında geliştirilmiş kapsamlı bir yapay zeka sistemidir. Temel amaç; sisteme yüklenen veya anlık kaydedilen video akışı üzerinden kişilerin yüz ifadelerini ve ses tonlarını eşzamanlı olarak analiz edip duygu durumu ve stres seviyelerini yüksek doğrulukla tespit etmektir. Sistem, elde edilen analiz sonuçlarını profesyonel bir şirket psikoloğu bakış açısıyla yorumlayarak nihai bir değerlendirme raporu sunacak şekilde tasarlanmıştır.

## Temel Özellikler ve Sistem Mimarisi

Sistem, modellerin eğitim aşamasından nihai raporun üretilmesine kadar ardışık ve birbirini besleyen bir yapıda tasarlanmıştır:

1. **Bağımsız Model Eğitimi:** Projenin temelini oluşturan derin öğrenme mimarileri (Görüntü için ConvNeXt ve Swin Transformer, ve Ses için özel ResNet18 ile eğitilmiş duygu analiz modeli) en yüksek doğruluğu sağlamak adına kendi veri setleri üzerinde ayrı ayrı eğitilmiştir.
2. **Görsel Analiz (Yüz ve Mimik İşleme):** Sisteme verilen video akışından anlık kareler (frameler) alınır. Bağımsız eğitilen görsel modeller, bu kareler üzerinden yüz hatlarını ve mimikleri analiz ederek anlık duygu durumu çıkarımı yapar.
3. **İşitsel Analiz (Ses Verisi İşleme):** Videodaki ses kanalı ayrıştırılır. Ses analiz modeli; kullanıcının ses tonunu, frekans değişimlerini ve vurgularını inceleyerek stres seviyesini ölçümler.
4. **Çok Modlu Entegrasyon (Multimodal Fusion):** Ayrı ayrı çalışan görüntü ve ses modellerinden elde edilen analiz sonuçları tek bir potada eritilir. Bu "fusion" işlemi, tek bir veri tipine bağlı kalmanın getirdiği hata payını düşürerek çok daha tutarlı ve güvenilir bir duygu/stres tahmini sağlar.
5. **Şeffaf ve Eşzamanlı Çıktı Ekranı:** Gradio arayüzü üzerinde sadece birleştirilmiş nihai sonuç gösterilmez; aynı zamanda ConvNeXt ve Swin Transformer modellerinin o anki kare için ayrı ayrı hangi duygu tahminlerini ürettiği, ses analizinin hangi duygu tahmininde bulunduğu da eşzamanlı olarak ekrana yazdırılır. Bu sayede modellerin bireysel kararları anlık ve şeffaf bir şekilde takip edilebilir.
6. **Şirket Psikoloğu Değerlendirme Modülü:** Analiz aşamasından çıkan çok modlu veriler son bir değerlendirme katmanından geçirilir. Sistem, elde edilen sonuçları profesyonel bir şirket psikoloğu bakış açısıyla yorumlayarak kullanıcının genel psikolojik durumunu özetleyen ve durum değerlendirmesi yapan kapsamlı bir metin raporu sunar.


## Kullanılan Teknolojiler ve Altyapı
* **Kullanıcı Arayüzü:** Gradio (Etkileşimli ve kullanıcı dostu arayüz)
* **Programlama Dili:** Python
* **Derin Öğrenme Mimarileri:** ConvNeXt, Swin Transformer
* **Veri İşleme:** Çok Modlu (Multimodal) Yapay Zeka Yaklaşımları

##  Model Ağırlık Dosyaları (.pth) Hakkında Önemli Not
GitHub dosya boyutu sınırlandırmaları (Maks. 100 MB) nedeniyle, projede kullanılan büyük boyutlu derin öğrenme modellerine ait ağırlık dosyaları (`convnext_best_model.pth` ve `swin_best_model.pth` ayrıca `ses_duygu_modeli_7sinif.pth` ve `yuz_tanima.xml` dosyaları da) harici bir bulut depolama alanına (Drive) yüklenmiştir. 
Sistemi yerel makinenizde (lokalde) çalıştırabilmek için lütfen öncelikle aşağıdaki bağlantıdan model dosyalarını indiriniz ve projenin ana dizinine yerleştiriniz:

🔗 **[Model Ağırlık Dosyalarını İndirmek İçin Tıklayın](https://drive.google.com/drive/folders/1lGkNrL3GXKKedn2EwWPqqs4u4lipVpnO?usp=sharing)**


##  Kurulum ve Çalıştırma
1. Bu depoyu bilgisayarınıza klonlayın veya `.zip` olarak indirin.
2. Yukarıdaki Drive linkinden büyük `.pth` dosyalarını indirip projenin ana klasörüne atın.
3. Gerekli kütüphaneleri kurun (Gradio, OpenCV, PyTorch, Librosa vb.).
4. `multimodal_app.py` veya arayüz dosyanızı çalıştırarak sistemi başlatın.
