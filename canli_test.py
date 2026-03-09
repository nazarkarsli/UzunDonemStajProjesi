import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class_names = ['Kizgin', 'Igrenme', 'Korku', 'Mutlu', 'Notr', 'Uzgun', 'Saskin'] # Kendi sırana göre düzelt!

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {device} kullaniliyor...")

#CONVNEXT 
print("ConvNeXt Modeli Yukleniyor...")
model_conv = models.convnext_tiny(weights=None)
model_conv.classifier[2] = nn.Linear(model_conv.classifier[2].in_features, len(class_names))

checkpoint_conv = torch.load('convnext_best_model.pth', map_location=device)
state_dict_conv = checkpoint_conv['model_state_dict'] if 'model_state_dict' in checkpoint_conv else checkpoint_conv
clean_state_dict_conv = {k.replace('module.', ''): v for k, v in state_dict_conv.items()}
model_conv.load_state_dict(clean_state_dict_conv)
model_conv = model_conv.to(device)
model_conv.eval()

#SWIN TRANSFORMER(TIMM UYUMLU)
import timm 
print("Swin Transformer Modeli (timm) Yukleniyor...")
model_swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=len(class_names))

checkpoint_swin = torch.load('swin_best_model.pth', map_location=device)
state_dict_swin = checkpoint_swin['model_state_dict'] if 'model_state_dict' in checkpoint_swin else checkpoint_swin

# Çift GPU (module.) temizliği
clean_state_dict_swin = {k.replace('module.', ''): v for k, v in state_dict_swin.items()}

model_swin.load_state_dict(clean_state_dict_swin)
model_swin = model_swin.to(device)
model_swin.eval()

print("Modeller basariyla yuklendi! Kamera aciliyor...")

# YÜZ TESPİTİ VE KAMERA DÖNGÜSÜ
import os
import urllib.request

# Türkçe karakterli uzun yolu atlatmak için dosyayı direkt projemizin yanına indiriyoruz
xml_path = 'yuz_tanima.xml'
if not os.path.exists(xml_path):
    print("Yuz tanima dosyasi indiriliyor ")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, xml_path)

face_cascade = cv2.CascadeClassifier(xml_path)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Yüzü mavi kutu içine al
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        face_img = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)
        
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            # ConvNeXt'in fikri
            out_conv = model_conv(input_tensor)
            _, pred_conv = torch.max(out_conv, 1)
            emotion_conv = class_names[pred_conv.item()]

            # Swin'in fikri
            out_swin = model_swin(input_tensor)
            _, pred_swin = torch.max(out_swin, 1)
            emotion_swin = class_names[pred_swin.item()]

        # Tahminleri yüzün üstüne yaz (Yeşil olan ConvNeXt, Sarı olan Swin)
        cv2.putText(frame, f"ConvNeXt: {emotion_conv}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Swin: {emotion_swin}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Çıkmak 'q' tusuna basın", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()