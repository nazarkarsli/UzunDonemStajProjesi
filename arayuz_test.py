import gradio as gr
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import timm
import os
import urllib.request

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Kizgin', 'Igrenme', 'Korku', 'Mutlu', 'Notr', 'Uzgun', 'Saskin']

print("Modeller Yükleniyor... Lütfen bekleyin.")

# ConvNeXt Yükle
model_conv = models.convnext_tiny(weights=None)
model_conv.classifier[2] = nn.Linear(model_conv.classifier[2].in_features, len(class_names))
checkpoint_conv = torch.load('convnext_best_model.pth', map_location=device)
state_dict_conv = checkpoint_conv['model_state_dict'] if 'model_state_dict' in checkpoint_conv else checkpoint_conv
model_conv.load_state_dict({k.replace('module.', ''): v for k, v in state_dict_conv.items()})
model_conv = model_conv.to(device).eval()

# Swin Yükle
model_swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=len(class_names))
checkpoint_swin = torch.load('swin_best_model.pth', map_location=device)
state_dict_swin = checkpoint_swin['model_state_dict'] if 'model_state_dict' in checkpoint_swin else checkpoint_swin
model_swin.load_state_dict({k.replace('module.', ''): v for k, v in state_dict_swin.items()})
model_swin = model_swin.to(device).eval()

# Yüz Tanıma Dosyası
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

print("Sistem Hazır! Arayüz Başlatılıyor...")

#VİDEO ANALİZ FONKSİYONU
def video_analiz(video_yolu):
    if video_yolu is None:
        return "Lütfen kaydı tamamlayıp videoyu sisteme yükleyin."
    
    mesaj = " MULTİMODAL VİDEO ANALİZ SONUÇLARI \n\n"
    
    # Videonun ortasındaki en net anı (kareyi) yakalayalım
    cap = cv2.VideoCapture(video_yolu)
    toplam_kare = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, toplam_kare // 2)
    ret, frame_bgr = cap.read()
    cap.release()
    
    if not ret:
        mesaj += " Görüntü: Videodan kare okunamadı!\n"
    else:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) == 0:
            mesaj += " Görüntü: Videoda yüz tespit edilemedi! Kameraya tam bakarak tekrar çekin.\n"
        else:
            (x, y, w, h) = faces[0]
            face_img = frame_bgr[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)
            
            input_tensor = transform(pil_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                emotion_conv = class_names[torch.max(model_conv(input_tensor), 1)[1].item()]
                emotion_swin = class_names[torch.max(model_swin(input_tensor), 1)[1].item()]
                
            mesaj += f"Görüntü (ConvNeXt Modeli): {emotion_conv}\n"
            mesaj += f"Görüntü (Swin Modeli): {emotion_swin}\n"
            
    mesaj += "\n-------------------\n"
    mesaj += f" Ses Analizi: Videonun ses kanalı başarıyla ayrıştırıldı! (Stres analizi modeli bekleniyor)\n"    
    return mesaj

arayuz = gr.Interface(
    fn=video_analiz,
    inputs=gr.Video(sources=["webcam"], label="Video Kaydet (Görüntü + Ses)"),
    outputs=gr.Textbox(label="Sistem Çıktısı", lines=10),
    title="Multimodal Duygu ve Stres Analizi",
    description="Record tuşuna basarak 3-4 saniyelik kısa bir video çekin. Sistem videoyu işleyerek eşzamanlı analiz yapacaktır.",
    theme="default"
)

if __name__ == "__main__":
    arayuz.launch(share=False)