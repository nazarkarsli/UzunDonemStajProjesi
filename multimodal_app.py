import gradio as gr
import cv2 #karelere böleceğiz
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import timm
import librosa
import os
import urllib.request #xml
import traceback #kod çökerse hata hangi adımda
import warnings
import subprocess #pcdeki harici programları (FFmpeg) pythondan calıstırmak icin
import tempfile #sesi kaydetmesi için

from google import genai 
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Kizgin', 'Igrenme', 'Korku', 'Mutlu', 'Notr', 'Uzgun', 'Saskin']

print("Sistem hazirlaniyor, modeller yukleniyor...")

def load_convnext():
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(class_names))
    checkpoint = torch.load('convnext_best_model.pth', map_location=device) #kendi eğittiğimiz modeli yükledik burada
    state_dict = checkpoint.get('model_state_dict', checkpoint) 
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    return model.to(device).eval()

def load_swin():
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=len(class_names))
    checkpoint = torch.load('swin_best_model.pth', map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    return model.to(device).eval()

def load_resnet(): # Ses modelimiz (ResNet18). ResNet aslinda bir goruntu modelidir ama biz sesi spektrograma (resme) cevirip veriyoruz.
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    checkpoint = torch.load('ses_duygu_modeli_7sinif.pth', map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    return model.to(device).eval()

model_conv = load_convnext()
model_swin = load_swin()
audio_model = load_resnet()

xml_path = 'yuz_tanima.xml'
if not os.path.exists(xml_path):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml", xml_path)
face_cascade = cv2.CascadeClassifier(xml_path) 

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def analyze_image(video_path):
    cap = cv2.VideoCapture(video_path)
    toplam_kare = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if toplam_kare > 0: 
        cap.set(cv2.CAP_PROP_POS_FRAMES, toplam_kare // 2)
    
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    cap.release()
    
    if not ret: 
        return "Bulunamadi", "Bulunamadi", "- Hata: Goruntu okunamadi."

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0: 
        return "Bulunamadi", "Bulunamadi", "- Hata: Yuz tespit edilemedi."
    
    (x, y, w, h) = faces[0]
    face_rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB) # OpenCV resimleri BGR (Mavi-Yesil-Kirmizi) okur. PyTorch ise RGB ister. Rengini ceviriyoruz.
    input_tensor = transform(Image.fromarray(face_rgb)).unsqueeze(0).to(device) #modelin anlayabileceği formata soktuk
    
    with torch.no_grad():
        res_conv = class_names[torch.max(model_conv(input_tensor), 1)[1].item()]
        res_swin = class_names[torch.max(model_swin(input_tensor), 1)[1].item()]
        # Modellere resmi veriyoruz. Cikan olasiliklar (1x7 matris) icinden en buyuk olanin indexini (argmax) alip class_names'ten Turkcesini buluyoruz.
    return res_conv, res_swin, ""

def analyze_audio(video_path): # Windows klasor hatalarini asmak icin gecici dosyamizi "Temp" icine tanimliyoruz.
    temp_wav = os.path.join(tempfile.gettempdir(), "gecici_ses_nazar.wav").replace("\\", "/")
    
    try:
        # Alt islem olarak (subprocess) sisteme gomulu FFmpeg'i cagiriyoruz.
        # -i video_path: Girdimiz.
        # -vn: Videoyu at (sadece ses kalsin).
        # -ar 22050: Sesin ornekleme hizi (Librosa'nin standardi).
        islem = subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1", temp_wav], 
                       capture_output=True, text=True)
        # FFmpeg gercekten dosyayi uretti mi diye guvenlik kontrolu yapiyoruz.
        if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) == 0:
            return "Bulunamadi", "- Hata: Tarayicin / Gradio SESSİZ (mute) video kaydediyor! Izinleri kontrol et."
        # Sesi (Zaman serisi dizisi olarak) bellege yukluyoruz.   
        y_audio, sr = librosa.load(temp_wav, sr=22050)
        os.remove(temp_wav)
        
        if len(y_audio) == 0: 
            return "Bulunamadi", "- Hata: Video icinde ses dalgasi yok."
        # Sesi gorsel bir haritaya (Mel-Spektrogram) ceviriyoruz ki goruntu modeli olan ResNet18 bunu anlayabilsin.
        S = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)

        # Spektrogram matrisini (0-255 arasi) resim formatina donusturup ayni yuzde yaptigimiz gibi Tensor'e ceviriyoruz.
        img_s = (S_db - S_db.min()) / (S_db.max() - S_db.min()) * 255
        img_s_tensor = transform(Image.fromarray(img_s.astype(np.uint8)).convert('RGB')).unsqueeze(0).to(device)
        # Sesi (Spektrogram resmini) ResNet18 modeline sokup tahmini aliyoruz.
        with torch.no_grad():
            res_audio = class_names[torch.max(audio_model(img_s_tensor), 1)[1].item()]
        return res_audio, ""
    except Exception as e:
        return "Bulunamadi", f"- Hata: Ses analiz edilemedi. ({str(e)})"

def gemini_degerlendirmesi_al(conv, swin, ses):
    try:
        client = genai.Client(api_key="AIzaSyB-cnjbLsmnXbBwVNepepDNu1IzlEdSlQA")
        prompt = f"""
        Sen bir is yeri psikologu ve insan kaynaklari uzmanisin. 
        Bir calisanin yapay zeka ile yapilan anlik duygu analizi sonuclari sunlar:
        - Yuz Ifadesi (ConvNeXt): {conv}
        - Yuz Ifadesi (Swin): {swin}
        - Ses Tonu (ResNet18): {ses}
        
        Lutfen bu uc veriyi degerlendirerek, calisanin mevcut duygu durumu hakkinda 5 cumlelik profesyonel bir degerlendirme ve kisa bir eylem tavsiyesi yaz.
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        # LLM cokerse bile sistemin geneli calismaya devam eder!
        return f"Yapay Zeka Baglanti Hatasi: {str(e)}\n(Merak etmeyin, ana modeller calismaya devam ediyor!)"
    
def video_analiz(video_yolu):
    if not video_yolu: 
        return "HATA: Video yuklenmedi.", "Degerlendirme yapilamadi."
    video_yolu = str(video_yolu.get('video', video_yolu) if isinstance(video_yolu, dict) else (video_yolu[0] if isinstance(video_yolu, tuple) else video_yolu))
    video_yolu = os.path.abspath(video_yolu).replace("\\", "/") # Windows'un ters slash (\) probleminden kacmak icin absolute (kesin) yolu bulup slashlari (/) duzeltiyoruz.
    
    conv_sonuc, swin_sonuc, img_hata = analyze_image(video_yolu)
    ses_sonuc, ses_hata = analyze_audio(video_yolu)
    
    # Ensemble (Coklu Model Karari) Mantigi:
    # 3 modelin cikardigi sonuclari bir listeye koyuyoruz. En cok tekrar eden (max + count) bizim kesin kararimiz oluyor (Voting Sistemi).
    gecerli_tahminler = [t for t in [conv_sonuc, swin_sonuc, ses_sonuc] if t in class_names]
    genel_tahmin = max(set(gecerli_tahminler), key=gecerli_tahminler.count) if gecerli_tahminler else "GECERSIZ"

    rapor = (
        f"-----------------------------------------\n"
        f"MULTIMODAL ANALIZ SONUCLARI\n"
        f"-----------------------------------------\n\n"
        f"GORUNTU MODELLERI:\n"
        f"- ConvNeXt Karari: {conv_sonuc}\n"
        f"- Swin Transformer Karari: {swin_sonuc}\n"
        f"{img_hata}\n"
        f"SES MODELI:\n"
        f"- ResNet18 Karari: {ses_sonuc}\n"
        f"{ses_hata}\n"
        f"-----------------------------------------\n"
        f"GENEL TAHMIN: {genel_tahmin.upper()}\n"
        f"-----------------------------------------"
    )
    temiz_rapor = rapor.replace("\n\n\n", "\n\n") # Eger hata yoksa, metindeki cirkin bos satirlari siliyoruz
    
    # AI DEGERLENDIRMESINI ALIYORUZ
    llm_raporu = gemini_degerlendirmesi_al(conv_sonuc, swin_sonuc, ses_sonuc)
    
    return temiz_rapor, llm_raporu

#GRADIO ARAYUZU
arayuz = gr.Interface(
    fn=video_analiz,
    #Sesi zorla almak icin include_audio=True eklendi
    # include_audio=True: Tarayiciyi, mikrofonu da zorla videonun icine kaydetmesi icin yonlendirir.
    # format="mp4": Sorunlu WebM yerine standart MP4 cikarmaya calisir.
    inputs=gr.Video(sources=["webcam"], include_audio=True, format="mp4", label="Video Kaydet (Goruntu + Ses)"),
    # DIKKAT: Outputs kismini 2 kutu olacak sekilde listeye cevirdik!
    outputs=[
        gr.Textbox(label="Sistem Ciktisi", lines=12),
        gr.Textbox(label="Yapay Zeka (Gemini) Raporu", lines=5)
    ],
    title="Multimodal Duygu ve Stres Analizi",
    description="Record tusuna basarak kisa bir video cekin. Goruntu ve ses eszamanli analiz edilecektir.",
)

if __name__ == "__main__":
    arayuz.launch(share=False)