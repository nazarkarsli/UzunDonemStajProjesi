# Çok Modlu Duygu ve Stres Analizi Sistemi

##  Proje Hakkında
Bu proje, Bilgisayar Mühendisliği 7+1 uzun dönem staj uygulaması kapsamında geliştirilmiş kapsamlı bir yapay zeka sistemidir.
Temel amaç; sisteme yüklenen veya anlık kaydedilen video akışı üzerinden kişilerin yüz ifadelerini ve ses tonlarını eşzamanlı olarak analiz edip duygu durumu ve stres seviyelerini yüksek doğrulukla tespit etmektir.
Sistem, elde edilen analiz sonuçlarını profesyonel bir şirket psikoloğu bakış açısıyla yorumlayarak nihai bir değerlendirme raporu sunacak şekilde tasarlanmıştır.

## Temel Özellikler ve Sistem Mimarisi
Sistem, modellerin eğitim aşamasından nihai raporun üretilmesine kadar ardışık ve birbirini besleyen bir yapıda tasarlanmıştır:

1. **Bağımsız Model Eğitimi:** Projenin temelini oluşturan derin öğrenme mimarileri (Görüntü için ConvNeXt ve Swin Transformer; Ses için ResNet18 ile eğitilmiş özel duygu analiz modeli) en yüksek doğruluğu sağlamak adına kendi veri setleri üzerinde ayrı ayrı eğitilmiştir.
2. **Görsel Analiz (Yüz ve Mimik İşleme):** Sisteme verilen video akışından anlık kareler (frameler) alınır. Bağımsız eğitilen görsel modeller, bu kareler üzerinden yüz hatlarını ve mimikleri analiz ederek anlık duygu durumu çıkarımı yapar.
3. **İşitsel Analiz (Ses Verisi İşleme):** Videodaki ses kanalı ayrıştırılır. Ses analiz modeli; kullanıcının ses tonunu, frekans değişimlerini ve vurgularını inceleyerek stres seviyesini ölçümler.
4. **Çok Modlu Entegrasyon (Multimodal Fusion):** Ayrı ayrı çalışan görüntü ve ses modellerinden elde edilen analiz sonuçları tek bir potada eritilir. Bu birleştirme işlemi, tek bir veri tipine bağlı kalmanın getirdiği hata payını düşürerek çok daha tutarlı ve güvenilir bir duygu/stres tahmini sağlar.
5. **Şeffaf ve Eşzamanlı Çıktı Ekranı:** Gradio arayüzü üzerinde sadece birleştirilmiş nihai sonuç gösterilmez; aynı zamanda ConvNeXt, Swin Transformer ve ses analiz modellerinin o anki tahminleri eşzamanlı olarak ekrana yazdırılır. Bu sayede modellerin bireysel kararları şeffaf bir şekilde takip edilebilir.
6. **Şirket Psikoloğu Değerlendirme Modülü:** Analiz aşamasından çıkan çok modlu veriler son bir değerlendirme katmanından geçirilir. Sistem, elde edilen sonuçları profesyonel bir şirket psikoloğu bakış açısıyla yorumlayarak kullanıcının genel psikolojik durumunu özetleyen ve tavsiye veren kapsamlı bir metin raporu sunar.


## Geliştirme Süreci ve Teknik Analiz Detayları

Bu proje, standart bir sınıflandırma probleminin ötesine geçerek; farklı veri tiplerinin (görüntü ve ses) ve farklı derin öğrenme mimarilerinin (CNN ve Transformer) bütüncül bir yaklaşımla entegre edilmesi sürecini kapsar. Staj dönemi boyunca izlenen geliştirme adımları ve mimari kararlar aşağıda detaylandırılmıştır:

### 1. Görsel Analiz Modelleri: Swin Transformer ve ConvNeXt Entegrasyonu
Projenin görsel boyutunda, yüz ifadelerinden duygu tahmini yapmak amacıyla tek bir modele bağlı kalmak yerine, birbirinin zayıf yönlerini kapatan iki farklı derin öğrenme mimarisi tercih edilmiştir. Bu mimari seçimlerin arkasındaki temel mühendislik yaklaşımı şu şekildedir:

**Neden Swin Transformer? (Global Bağlam ve Dikkat Mekanizması):** İlk aşamada, yüzün genel anatomisini analiz etmek için bir Vision Transformer (ViT) varyantı olan Swin Transformer mimarisi kullanılmıştır. Model, sahip olduğu "Shifted Window" (Kaydırılmış Pencere) mekanizması ve dikkat (attention) katmanları sayesinde yüzdeki organlar arası uzaklık ilişkisini (örneğin kaş ve ağız hareketlerinin bütünlüğünü) çok iyi analiz eder. Ancak, bu model eğitim sürecinde doğrulama (validation) başarısı %70.52 seviyesinde doygunluğa ulaşmış ve sistemde bir performans sınırı (bottleneck) yaşanmıştır.

**Neden ConvNeXt? (Yerel Özellikler):** Swin Transformer'ın ulaştığı bu sınırı aşmak amacıyla sisteme dahil edilen ConvNeXt, bireysel testlerde %85.39 gibi yüksek bir doğruluk oranına ulaşmıştır. Bu modelin seçilme amacı; yüzdeki ince dokusal değişimleri ve anlık mikro mimikleri (yerel özellikleri) Swin'in global bakış açısına göre çok daha keskin bir şekilde yakalayarak sistemin toplam başarısını yukarı çekmesidir.

**Fusion (Birliktelik) Mantığı:** Swin Transformer'ın "bütünü gören (global)" yapısı ile ConvNeXt'in "detaya inen (local)" yapısı birleştirilmiştir. Bu mimari işbirliği sayesinde, tek bir modelin zayıf kalabileceği ters ışık, farklı yüz açıları veya düşük çözünürlük gibi dezavantajlı durumlarda sistemin çok daha kararlı (robust) çalışması sağlanmıştır.

**Model Ağırlıklarının (Weights) Optimizasyonu:** Her iki modelin de eğitim sürecinde doğrulama setinde en yüksek performansı gösteren ideal ağırlıkları tespit edilmiş ve son kullanıcı aşamasında modelin tekrar eğitilmesine gerek kalmadan doğrudan kullanılabilmesi için .pth formatında (best model weights) kaydedilmiştir. OpenCV ile videolardan anlık olarak tespit edilen yüzler, bu optimize edilmiş ağırlıklar üzerinden eşzamanlı geçirilerek canlı çıkarım (inference) yapılmaktadır.

### 2. İşitsel Analiz Modeli: Mel-Spektrogram ve ResNet18
Görsel verinin yetersiz kaldığı durumlarda veya duygu durumunun çok modlu olarak doğrulanması amacıyla ses verisi analiz sürecine eklenmiştir:
* Ham ses dalgaları üzerinden doğrudan analiz yapmak yerine, sinyal işleme teknikleri kullanılarak ses verisi `Librosa` kütüphanesi ile **Mel-Spektrogram** haritalarına dönüştürülmüştür.
   Bu sayede insan sesindeki frekans kırılmaları ve enerji seviyeleri görsel bir matris haline getirilmiştir.
* Elde edilen bu 2 boyutlu(2D) haritalar, özellik çıkarımında rüştünü ispatlamış ResNet18 mimarisine beslenmiştir. Ağın derin yapısı, sesteki anlık paternleri inceleyerek stres seviyesi ve duygu sınıflandırmasını gerçekleştirir. Ses modelinin de en başarılı eğitim ağırlığı ayrı bir `.pth` dosyası olarak kaydedilmiştir.

### 3. Çok Modlu Birleştirme ve Sistem Uygulaması (Gradio)
Bağımsız olarak eğitilen ve yüksek doğruluk oranlarına ulaşan bu modeller, etkileşimli bir kullanıcı arayüzü sunan `Gradio` platformu üzerinde, `multimodal_app.py` dosyası aracılığıyla birleştirilmiştir. 
* **Doğrudan Çıkarım (Inference) Mantığı:** Sistem başlatıldığında modelleri her seferinde sıfırdan eğitmek veya hesaplamak yerine, eğitim aşamasında elde edilip kaydedilen bu en başarılı ağırlık dosyaları (`.pth`) doğrudan sisteme yüklenir. Bu mühendislik yaklaşımı sayesinde sistem, kullanıcıdan alınan video akışını arka planda gecikmesiz ayrıştırarak eşzamanlı olarak görsel ve işitsel modellere iletebilir.
* Modellerin ürettiği bireysel tahminler, algoritmaların şeffaflığı ilkesi gereği arayüzde ayrı ayrı listelenir ve nihayetinde bu veriler birleştirilerek tek bir duygu/stres tahmini elde edilir. 

---

## Kullanılan Teknolojiler ve Kütüphaneler
* **Kullanıcı Arayüzü:** Gradio
* **Programlama Dili:** Python
* **Derin Öğrenme Mimarileri:** ConvNeXt, Swin Transformer, ResNet18
* **Görüntü ve Ses İşleme:** OpenCV (Yüz Tespiti), Librosa (Sinyal İşleme)
* **Kütüphaneler:** PyTorch, Torchvision

##  Kurulum ve Çalıştırma Yönergesi
Sistemi yerel ortamda çalıştırmak için; doğrudan README dosyasında yer alan **DRIVE Linki üzerinden** `.pth` uzantılı model ağırlıklarını indirip projenin ana dizinine yerleştirmeniz ve ardından `multimodal_app.py` dosyasını çalıştırmanız yeterlidir. 

*(Model eğitim süreçlerini içeren defterler olan `Swin_Transformer_Kodu_C.ipynb` ve `ConvNeXt_Kodu_K.ipynb`, projenin şeffaflığı açısından `model_egitim_kodlari` klasörüne ayrıca eklenmiştir. Ayrı ayrı eğitilen bu modellerin çok modlu entegrasyonu ise `multimodal_app.py` dosyası üzerinden sağlanmaktadır.)*

🔗 **[Model Ağırlık Dosyalarını İndirmek İçin Tıklayın](https://drive.google.com/drive/folders/1lGkNrL3GXKKedn2EwWPqqs4u4lipVpnO?usp=sharing)**

**Adımlar:**
1. Bu depoyu bilgisayarınıza klonlayın veya `.zip` olarak indirin.
2. Yukarıdaki Drive linkinden büyük `.pth` dosyalarını (`convnext_best_model.pth`, `swin_best_model.pth`, `ses_duygu_modeli_7sinif.pth`) ve `yuz_tanima.xml` dosyasını indirip projenin ana dizinine (klasörüne) atın.
3. Gerekli kütüphaneleri kurun.
4. `multimodal_app.py` dosyasını çalıştırarak sistemi başlatın.
